import os
import struct
import argparse
import traceback
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_pcd_point_format(header_data: dict):
    """
    Dynamically determines the struct format, byte size, and ring index from the PCD header.
    Returns (struct_format, point_byte_size, ring_field_index) or (None, None, None) on failure.
    """
    try:
        fields = header_data['FIELDS']
        sizes = [int(s) for s in header_data['SIZE']]
        types = header_data['TYPE']

        try:
            ring_field_index = fields.index('ring')
        except ValueError:
            # This is a critical error for the script's logic, so we print and return None.
            print(f"  [ERROR] 'ring' field not found in PCD header's FIELDS line.")
            return None, None, None

        # Map PCD types to python struct format characters
        type_map = {
            ('F', 4): 'f', ('F', 8): 'd',
            ('I', 1): 'b', ('I', 2): 'h', ('I', 4): 'i', ('I', 8): 'q',
            ('U', 1): 'B', ('U', 2): 'H', ('U', 4): 'I', ('U', 8): 'Q',
        }

        format_str = '<'  # Little-endian
        for i in range(len(fields)):
            char = type_map.get((types[i], sizes[i]))
            if char is None:
                print(f"  [ERROR] Unsupported TYPE/SIZE combination: {types[i]}/{sizes[i]}")
                return None, None, None
            format_str += char

        point_byte_size = struct.calcsize(format_str)
        return format_str, point_byte_size, ring_field_index

    except (KeyError, IndexError, ValueError) as e:
        print(f"  [ERROR] Failed to parse header for dynamic format. Error: {e}")
        return None, None, None

def process_pcd_file(pcd_path: Path, output_dir: Path, ring_ids_to_filter: set, inverse_select: bool):
    """
    Reads a single binary PCD file, filters its points based on the 'ring' value,
    and writes the result to a new PCD file in the specified output directory.
    """
    print(f"--- Processing file: {pcd_path.name} ---")
    try:
        with open(pcd_path, 'rb') as f:
            # 1. Parse the PCD header
            header_lines = []
            header_data = {}
            while True:
                line_bytes = f.readline()
                header_lines.append(line_bytes)
                try:
                    decoded_line = line_bytes.decode('ascii').strip()
                except UnicodeDecodeError:
                    print(f"  [ERROR] Could not decode header line in {pcd_path.name}. Skipping file.")
                    return f"FAIL: Unicode error in header of {pcd_path.name}"

                parts = decoded_line.split()
                if len(parts) >= 2:
                    header_data[parts[0]] = parts[1:]
                if decoded_line.startswith('DATA'):
                    break
            
            if header_data.get('DATA', ['ascii'])[0].lower() != 'binary':
                print(f"  [ERROR] Script only supports 'DATA binary' format in {pcd_path.name}. Skipping.")
                return f"FAIL: Not binary format in {pcd_path.name}"

            num_points = int(header_data.get('POINTS', [0])[0])
            if num_points == 0:
                print(f"  [INFO] File {pcd_path.name} contains 0 points. Skipping.")
                return f"SUCCESS: {pcd_path.name} (0 points)"

            # 2. Dynamically determine point structure from header
            point_struct_format, point_byte_size, ring_index = get_pcd_point_format(header_data)
            if not point_struct_format:
                print(f"  [ERROR] Could not determine point structure for '{pcd_path.name}'. Skipping.")
                return f"FAIL: Bad point structure in {pcd_path.name}"

            # 3. Read all points and filter them based on ring ID
            filtered_points_bytes = []
            for _ in range(num_points):
                point_bytes = f.read(point_byte_size)
                if len(point_bytes) < point_byte_size:
                    print("  [WARNING] Reached end of file unexpectedly.")
                    break
                
                unpacked_data = struct.unpack(point_struct_format, point_bytes)
                ring_value = unpacked_data[ring_index]
                
                # Apply filtering logic
                is_in_set = ring_value in ring_ids_to_filter
                if (is_in_set and not inverse_select) or (not is_in_set and inverse_select):
                    filtered_points_bytes.append(point_bytes)

            # 4. Write the new filtered PCD file
            num_filtered_points = len(filtered_points_bytes)
            if num_filtered_points == 0:
                print(f"  [INFO] No points matched the filter criteria for {pcd_path.name}.")
                return f"SUCCESS: {pcd_path.name} (0 points filtered)"

            # Create new header
            new_header_lines = []
            for line in header_lines:
                decoded_line = line.decode('ascii').strip()
                if decoded_line.startswith('WIDTH') or decoded_line.startswith('POINTS'):
                    key = decoded_line.split()[0]
                    new_header_lines.append(f"{key} {num_filtered_points}\n".encode('ascii'))
                else:
                    new_header_lines.append(line)
            new_header = b''.join(new_header_lines)

            # Write the new file
            output_filepath = output_dir / pcd_path.name
            with open(output_filepath, 'wb') as out_f:
                out_f.write(new_header)
                out_f.write(b''.join(filtered_points_bytes))

            print(f"  -> Created '{output_filepath.name}' with {num_filtered_points} points.")
            return f"SUCCESS: {pcd_path.name}"

    except Exception as e:
        print(f"  [FATAL ERROR] An unexpected error occurred while processing {pcd_path.name}: {e}")
        traceback.print_exc()
        return f"FAIL: Exception in {pcd_path.name}"

def main():
    """
    Main function to find, filter, and process PCD files in parallel.
    """
    parser = argparse.ArgumentParser(
        description="Filter points from PCD files based on their 'ring' ID and process them in parallel.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input_folder",
        type=str,
        help="Path to the folder containing the input .pcd files."
    )
    parser.add_argument(
        "-s", "--suffix",
        type=str,
        default="_filtered",
        help="Suffix to append to the input folder name to create the output folder.\n(default: _filtered)"
    )
    parser.add_argument(
        "-r", "--rings",
        type=int,
        nargs='+',
        required=True,
        help="A list of integer ring IDs to select (or exclude with --inverse)."
    )
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="If set, selects all points EXCEPT those with the given ring IDs."
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=0,
        help="Number of parallel processes to use.\nDefault is half the number of CPU cores. Set to 1 for sequential processing."
    )
    args = parser.parse_args()

    # --- Setup and Validation ---
    input_dir = Path(args.input_folder)
    if not input_dir.is_dir():
        print(f"Error: Input folder not found at '{input_dir}'")
        return

    output_dir = input_dir.with_name(f"{input_dir.name}{args.suffix}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{output_dir}'. Error: {e}")
        return

    num_jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1) // 2
    if num_jobs == 0: num_jobs = 1

    ring_ids_set = set(args.rings)
    pcd_files = sorted(list(input_dir.glob('*.pcd')))
    total_files = len(pcd_files)

    if not pcd_files:
        print("No .pcd files found in the input folder.")
        return

    # --- Print Summary ---
    print("--- PCD Filtering Configuration ---")
    print(f"Input folder:  '{input_dir}'")
    print(f"Output folder: '{output_dir}'")
    print(f"Ring IDs:      {sorted(list(ring_ids_set))}")
    print(f"Selection mode: {'INVERSE (exclude rings)' if args.inverse else 'NORMAL (include rings)'}")
    print(f"Parallel jobs: {num_jobs}")
    print(f"Files to process: {total_files}")
    print("---------------------------------")


    # --- Parallel Processing ---
    # Use functools.partial to create a worker function with fixed arguments
    worker_func = partial(
        process_pcd_file,
        output_dir=output_dir,
        ring_ids_to_filter=ring_ids_set,
        inverse_select=args.inverse
    )

    success_count = 0
    fail_count = 0
    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        # Submit all files to the executor
        futures = {executor.submit(worker_func, pcd_file): pcd_file for pcd_file in pcd_files}
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result.startswith("SUCCESS"):
                success_count += 1
            else:
                fail_count += 1
            print(f"Progress: {i}/{total_files} files completed.")


    print("\n--- Process complete. ---")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed or skipped:    {fail_count} files")

if __name__ == "__main__":
    main()