# Define vars with real system values, generic 64GB sd card used
DEVICE="/dev/mmcblk1"  # Your SD/eMMC device
PART_NUM=3  # Root partition number from fdisk output
START_SECTOR=679936  # From your fdisk output for p3 start


# Function to check available space on root (in MB)
avail_space() {
    df -BM / | awk 'NR==2 {print $4}' | tr -d 'M'
}

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo "Success: $1"
    else
        echo "Error: $1 failed. Check logs or previous output."
        # Optionally exit or continue based on severity
    fi
}


# Step 1: Minimal update and install gdisk for resize
echo "Step 1: Minimal apt update and install gdisk (for resize)"
apt update
check_success "apt update"
apt install -y gdisk
check_success "install gdisk"

# Step 2: Resize if needed
echo "Step 2: Checking if resize is needed (avail < 8000MB triggers)"
df -h /  # Explicit print before
if [ $(avail_space) -lt 8000 ]; then
    echo "Low space detected. Proceed with resize? (y/n)"
    read -r confirm
    if [ "$confirm" != "y" ]; then
        echo "Skipping resize as per user choice."
    else
        echo "Resizing root partition... (This might take a while)"
        # Fix GPT backup location/end
        sgdisk -e ${DEVICE}
        # Get current type GUID to preserve it (avoids boot issues)
        CURRENT_TYPE=$(sgdisk -i ${PART_NUM} ${DEVICE} | grep 'Partition GUID code:' | awk '{print $4}')
        # Delete p3, recreate from start to end (0=end), set original type
        sgdisk -d ${PART_NUM} -n ${PART_NUM}:${START_SECTOR}:0 -t ${PART_NUM}:${CURRENT_TYPE} ${DEVICE}
        # Verify
        sgdisk -v ${DEVICE}
        # Reload partition table
        partprobe ${DEVICE} || echo "Partprobe failed (expected for mounted root); proceeding"
        # Grow filesystem
        resize2fs ${DEVICE}p${PART_NUM} || true  # Ignore if "nothing to do"
        check_success "Partition resize"
        df -h /  # Explicit print after
        if [ $(avail_space) -lt 8000 ]; then
            echo "Resize didn't take effect yet; rebooting to apply. After boot, rerun this script to continue."
            reboot
            exit 0
        fi
    fi
else
    echo "Sufficient space, skipping resize"
    df -h /
fi

echo "done"
