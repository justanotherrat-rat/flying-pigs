#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Step 1: Running the preprocessing Python script..."
python3 losGenerationBands.py

echo "Step 2: Running the ns-3 simulation..."
# Use ./ns3 run (for newer ns-3 versions) or ./waf --run (for older versions)
./ns3 run "scratch/uav_mesh_eval --dir=ns3_export --useUav=1 --out=with_uav"

echo "Workflow completed successfully!"