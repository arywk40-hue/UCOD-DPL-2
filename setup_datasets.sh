#!/bin/bash

# Setup script for UCOD-DPL-2 datasets
# This script creates the necessary directory structure and symlinks

echo "=== UCOD-DPL-2 Dataset Setup ==="
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if data/RefCOD exists
if [ -d "data/RefCOD" ]; then
    echo "✓ Found data/RefCOD directory"
    
    # Check for required dataset folders
    if [ -d "data/RefCOD/TR-CAMO/im" ] && [ -d "data/RefCOD/TR-COD10K/im" ]; then
        echo "✓ Found TR-CAMO and TR-COD10K datasets"
        
        # Count images
        TR_CAMO_COUNT=$(find data/RefCOD/TR-CAMO/im -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
        TR_COD10K_COUNT=$(find data/RefCOD/TR-COD10K/im -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
        
        echo "  - TR-CAMO images: $TR_CAMO_COUNT"
        echo "  - TR-COD10K images: $TR_COD10K_COUNT"
        echo ""
        
        # Create symlink
        if [ -L "datasets" ]; then
            echo "✓ Symlink 'datasets' already exists"
        elif [ -d "datasets" ]; then
            echo "⚠ 'datasets' exists as a directory (not a symlink)"
            echo "  Consider removing it and running this script again"
        else
            echo "Creating symlink: datasets -> data"
            ln -s data datasets
            echo "✓ Symlink created successfully"
        fi
        
        echo ""
        echo "=== Setup Complete ==="
        echo ""
        echo "You can now run training with:"
        echo "  python train.py --config configs/uscod/ADF_dinov2.py"
        
    else
        echo "✗ ERROR: Required datasets not found in data/RefCOD/"
        echo ""
        echo "Expected structure:"
        echo "  data/RefCOD/TR-CAMO/im/     (training images)"
        echo "  data/RefCOD/TR-CAMO/gt/     (ground truth masks)"
        echo "  data/RefCOD/TR-COD10K/im/   (training images)"
        echo "  data/RefCOD/TR-COD10K/gt/   (ground truth masks)"
        echo ""
        echo "Please download and extract the datasets first."
    fi
    
else
    echo "✗ ERROR: data/RefCOD directory not found"
    echo ""
    echo "Searching for datasets in other locations..."
    
    # Search for datasets
    REFCOD_FOUND=$(find . -maxdepth 3 -type d -name "RefCOD" 2>/dev/null)
    TR_CAMO_FOUND=$(find . -maxdepth 4 -type d -name "TR-CAMO" 2>/dev/null)
    
    if [ -n "$REFCOD_FOUND" ]; then
        echo "Found RefCOD at: $REFCOD_FOUND"
    fi
    
    if [ -n "$TR_CAMO_FOUND" ]; then
        echo "Found TR-CAMO at: $TR_CAMO_FOUND"
    fi
    
    echo ""
    echo "Please organize your datasets as:"
    echo "  $SCRIPT_DIR/data/RefCOD/TR-CAMO/im/"
    echo "  $SCRIPT_DIR/data/RefCOD/TR-CAMO/gt/"
    echo "  $SCRIPT_DIR/data/RefCOD/TR-COD10K/im/"
    echo "  $SCRIPT_DIR/data/RefCOD/TR-COD10K/gt/"
    echo ""
    echo "Then run this script again."
fi
