# Radar Signal Processing (MATLAB)

## Prerequisites
- MATLAB installed and on your PATH
- Project folders accessible
- Set environment variable `CASCADE_SIGNAL_PROCESSING_CHAIN_MIMO` to the full path of `4chip_cascade_mimo_example`
  

## Quick Start
1. Navigate using the environment variable.
   ```matlab
   % In MATLAB:
   cd(getenv('CASCADE_SIGNAL_PROCESSING_CHAIN_MIMO'));
   ```
2. Run `add_paths.m` to set up dependencies.
   ```matlab
   add_paths;
   ```
3. Navigate to `cascade` and run the main processing script.
   ```matlab
   cd('cascade');
   cascade_mimo_signal_processing;
   ```

## Notes
- Ensure all required toolboxes are installed.
- Paths are relative to the project root.