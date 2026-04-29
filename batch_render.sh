#!/bin/bash

# Batch render script: all scenes × all backends

# Define scenes and rendering parameters
declare -A SCENES=(
    [ggx]="800 800 1024"
    [hdr]="800 800 1024"
    [furnace]="400 400 2048"
    [glass]="800 800 1024"
    [volume]="800 800 512"
    [sss]="800 800 512"
)

DEVICES=("cpu" "gpu")
DENOISE_OPTIONS=(0 1)  # 0=no denoise, 1=denoise
BUILD_DIR="./build"
RENDER_DIR="./renders"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure build exists
if [ ! -f "$BUILD_DIR/render" ]; then
    echo -e "${YELLOW}Building project...${NC}"
    cmake --build "$BUILD_DIR"
fi

total_renders=$((${#SCENES[@]} * ${#DEVICES[@]} * ${#DENOISE_OPTIONS[@]}))
current=0

echo -e "${BLUE}Starting batch render: ${total_renders} combinations${NC}"
echo "Scenes: ${!SCENES[@]}"
echo "Devices: ${DEVICES[@]}"
echo "Denoise: off / on"
echo ""

# Render all combinations
for scene in "${!SCENES[@]}"; do
    read -r width height spp <<< "${SCENES[$scene]}"
    
    for device in "${DEVICES[@]}"; do
        for denoise in "${DENOISE_OPTIONS[@]}"; do
            ((current++))
            
            # Generate output filename
            denoise_suffix=""
            denoise_flag=""
            if [ "$denoise" -eq 1 ]; then
                denoise_suffix="_denoised"
                denoise_flag="--denoise"
            fi
            output_file="${RENDER_DIR}/${scene}_${device}_${width}x${height}_${spp}spp${denoise_suffix}.ppm"
            
            # Skip if already rendered
            if [ -f "$output_file" ]; then
                echo -e "${YELLOW}[${current}/${total_renders}]${NC} SKIP: $scene ($device) - file exists"
                continue
            fi
            
            denoise_label="no-denoise"
            [ "$denoise" -eq 1 ] && denoise_label="denoised"
            
            echo -e "${BLUE}[${current}/${total_renders}]${NC} Rendering: ${GREEN}$scene${NC} ($device, $denoise_label) ${width}×${height} ${spp}spp"
            
            # Run render
            if $BUILD_DIR/render "$output_file" "$scene" \
                --width "$width" \
                --height "$height" \
                --spp "$spp" \
                --device "$device" \
                --no-preview \
                $denoise_flag; then
                echo -e "${GREEN}✓ Complete${NC}: $output_file"
            else
                echo -e "${YELLOW}✗ Failed${NC}: $scene ($device, $denoise_label)"
            fi
            echo ""
        done
    done
done

echo -e "${GREEN}Batch render complete!${NC}"
ls -lh "$RENDER_DIR" | tail -n +2 | awk '{print $9, "("$5")"}'
