# main.py

import argparse
import sys
import os

# Ensure the project root is in the Python path if main.py is in the root
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from simulation.run_simulation import run_simulation
    # Import get_scenario to potentially list available scenarios
    # This import also helps catch issues if scenarios.py itself has problems
    from simulation.scenarios import get_scenario
except ImportError as e:
    print(f"Error importing simulation modules in main.py: {e}")
    print("Please ensure you are running main.py from the project's root directory,")
    print("and all package __init__.py files are present if necessary.")
    # For debugging, print sys.path
    # print("Current sys.path:", sys.path)
    sys.exit(1)
except Exception as e: # Catch other potential errors during initial imports
    print(f"An unexpected error occurred during initial imports in main.py: {e}")
    sys.exit(1)


def list_available_scenarios():
    """
    Lists scenarios that are typically defined in scenarios.py.
    This function provides examples; for a dynamic list, scenarios.py
    would need to expose its defined scenario names.
    """
    # These names should match the 'if/elif scenario_name == ...' conditions in scenarios.py
    known_scenarios = [
        "strong_3agents_2tasks",
        "weak_2agents_2tasks_rendezvous",
        "weak_3agents_3tasks_exclusive",
        # Add other scenario names here as they are defined in scenarios.py
    ]
    print("Available scenarios (examples, check scenarios.py for actual definitions):")
    for name in known_scenarios:
        print(f"  - {name}")
    print("\nNote: Ensure these scenario names are correctly defined in 'simulation/scenarios.py'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Heterogeneous Spacecraft Collaboration Simulation.",
        formatter_class=argparse.RawTextHelpFormatter # Allows for better help text formatting
    )
    parser.add_argument(
        "scenario",
        type=str,
        nargs='?', # Makes the scenario argument optional if --list is used
        help="Name of the scenario to run (e.g., 'strong_3agents_2tasks').\n"
             "Use '--list' to see example scenario names."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the simulation (e.g., 42). Default: None."
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable dynamic visualization during the simulation."
    )
    parser.add_argument(
        "--save-anim",
        type=str,
        default=None,
        metavar="ANIM_FILE_PREFIX",
        help="Save the animation to a file. Provide a prefix (e.g., 'results/my_run').\n"
             "The scenario name, seed (if any), and .mp4 extension will be appended.\n"
             "Implies visualization is enabled unless --no-visualize is also set."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List example scenario names and exit."
    )

    args = parser.parse_args()

    if args.list:
        list_available_scenarios()
        sys.exit(0)

    if not args.scenario:
        parser.error("the following arguments are required: scenario. Use --list to see options.")
        sys.exit(1)


    scenario_name_to_run = args.scenario
    random_seed_to_use = args.seed
    enable_visualization = not args.no_visualize
    animation_file_prefix_arg = args.save_anim # This is the prefix from command line

    output_animation_full_prefix = None # This will be passed to run_simulation
    results_base_dir = "results" # Default directory for all outputs

    if enable_visualization and animation_file_prefix_arg:
        # If a path like "results/my_run" is given, use it.
        # If just "my_run" is given, prepend the default results_base_dir.
        if os.path.dirname(animation_file_prefix_arg): # Checks if a directory path is part of the prefix
            output_animation_full_prefix = animation_file_prefix_arg
        else:
            output_animation_full_prefix = os.path.join(results_base_dir, animation_file_prefix_arg)
        
        # Ensure the directory for the animation exists
        anim_dir = os.path.dirname(output_animation_full_prefix)
        if anim_dir and not os.path.exists(anim_dir):
            try:
                os.makedirs(anim_dir, exist_ok=True)
                print(f"Created output directory: {anim_dir}")
            except OSError as e:
                print(f"Warning: Could not create output directory {anim_dir}: {e}")
    elif enable_visualization: # Visualize but not save (prefix is None)
        output_animation_full_prefix = None
    else: # No visualization, no saving animation
        output_animation_full_prefix = None


    print(f"\nConfiguration:")
    print(f"  Scenario: '{scenario_name_to_run}'")
    print(f"  Seed: {random_seed_to_use if random_seed_to_use is not None else 'Not set (random)'}")
    print(f"  Visualize: {enable_visualization}")
    if output_animation_full_prefix:
        print(f"  Save Animation Prefix: '{output_animation_full_prefix}' (full name will include scenario/seed)")
    else:
        print(f"  Save Animation: No")


    try:
        print("\nAttempting to start simulation...")
        run_simulation(
            scenario_name=scenario_name_to_run,
            random_seed=random_seed_to_use,
            visualize=enable_visualization,
            animation_filename_prefix=output_animation_full_prefix # Corrected keyword argument
        )
    except ModuleNotFoundError as e: # Catch specific module errors if they persist from sub-modules
        print(f"\nCRITICAL ModuleNotFoundError: {e}")
        print("This often means a required Python file is missing or not in the correct path,")
        print("or there's an issue with relative/absolute imports within the project structure.")
        print("Ensure all .py files are in their designated package directories (e.g., common/, strong_communication/)")
        print("and that each package directory contains an __init__.py file.")
    except ValueError as e: # Catch ValueErrors, e.g., from scenario loading
        print(f"\nValueError during simulation setup or execution for scenario '{scenario_name_to_run}':")
        print(e)
        print("\nConsider using '--list' to check available scenario names and verify scenario definitions.")
    except Exception as e: # Catch any other unexpected errors
        print(f"\nAn unexpected error occurred during simulation for scenario '{scenario_name_to_run}':")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        # For more detailed debugging during development:
        import traceback
        print("\nFull Traceback:")
        traceback.print_exc()

    print("\n--- main.py Finished ---")