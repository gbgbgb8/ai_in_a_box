#!/bin/bash

# updateToPhi4.sh - Update AI in a Box from orca-mini to Phi-4-small
# https://github.com/gbgbgb8/ai_in_a_box

set -e  # Exit on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI in a Box - Upgrade to Phi-4 Assistant${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to get user confirmation
confirm() {
    read -p "$(echo -e "${YELLOW}$1 [y/n]: ${NC}")" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Operation canceled by user${NC}"
        return 1
    fi
    return 0
}

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Step 1: Check for USB drive and mount it for backups
echo -e "${BLUE}Step 1: Checking for USB drive for backups...${NC}"
USB_DRIVE=""
BACKUP_DIR=""

for dev in sda1 sdb1 sdc1; do
    if [ -b "/dev/$dev" ]; then
        echo -e "Found potential USB drive: /dev/$dev"
        if confirm "Would you like to use this drive for backups?"; then
            USB_DRIVE="/dev/$dev"
            break
        fi
    fi
done

if [ -z "$USB_DRIVE" ]; then
    echo -e "${YELLOW}No USB drive selected for backups.${NC}"
    if confirm "Continue without backing up to USB drive?"; then
        BACKUP_DIR="/home/ubuntu/ai_in_a_box_backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
    else
        exit 1
    fi
else
    echo -e "Mounting USB drive $USB_DRIVE"
    mkdir -p /media/usb
    mount "$USB_DRIVE" /media/usb
    BACKUP_DIR="/media/usb/ai_in_a_box_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    echo -e "${GREEN}USB drive mounted. Backups will be stored in $BACKUP_DIR${NC}"
fi

# Step 2: Backup original files
echo -e "\n${BLUE}Step 2: Backing up original files...${NC}"
if confirm "Backup original models and configuration?"; then
    echo "Creating backup directory structure..."
    mkdir -p "$BACKUP_DIR/downloaded"
    mkdir -p "$BACKUP_DIR/ai_in_a_box"
    
    echo "Backing up original orca model..."
    if [ -f "/home/ubuntu/downloaded/orca-mini-3b.ggmlv3.q4_0.bin" ]; then
        cp "/home/ubuntu/downloaded/orca-mini-3b.ggmlv3.q4_0.bin" "$BACKUP_DIR/downloaded/"
        echo -e "${GREEN}Orca model backed up successfully${NC}"
    else
        echo -e "${YELLOW}Orca model file not found, skipping...${NC}"
    fi
    
    echo "Backing up configuration files..."
    cp "/home/ubuntu/ai_in_a_box/llm_speaker.py" "$BACKUP_DIR/ai_in_a_box/"
    cp "/home/ubuntu/ai_in_a_box/run_chatty.sh" "$BACKUP_DIR/ai_in_a_box/"
    
    echo -e "${GREEN}Backup completed to $BACKUP_DIR${NC}"
fi

# Step 3: Download Phi-4-small model
echo -e "\n${BLUE}Step 3: Downloading Phi-4-small model...${NC}"
if confirm "Download Phi-4-small model (approx. 2GB)?"; then
    echo "Downloading Phi-4-small model..."
    mkdir -p /tmp/phi4-download
    cd /tmp/phi4-download
    wget --show-progress https://huggingface.co/TheBloke/phi-4-small-GGUF/resolve/main/phi-4-small.Q4_0.gguf
    
    echo "Moving model to downloaded directory..."
    mv phi-4-small.Q4_0.gguf /home/ubuntu/downloaded/
    
    echo -e "${GREEN}Phi-4-small model downloaded successfully${NC}"
fi

# Step 4: Update the llm_speaker.py configuration
echo -e "\n${BLUE}Step 4: Updating the llm_speaker.py configuration...${NC}"
if confirm "Update the model configuration file?"; then
    echo "Backing up the original llm_speaker.py..."
    cp /home/ubuntu/ai_in_a_box/llm_speaker.py /home/ubuntu/ai_in_a_box/llm_speaker.py.bak
    
    echo "Adding Phi-4-small model configuration..."
    sed -i '/model_param_dict = {/a \    "phi4-small"  : {\n        "file": "phi-4-small.Q4_0.gguf",\n        "ctx" : 2048,\n        "eps" : 1e-6,\n        "rfb" : 10000,\n        "pfx" : "### User: ",\n        "sfx" : "### Response:",\n        "init": "### System: You are an assistant that talks in a human-like "\\\n                "conversation style and provides useful, very brief, and concise "\\\n                "answers. Do not say what the user has said before."\n    },' /home/ubuntu/ai_in_a_box/llm_speaker.py
    
    echo -e "${GREEN}Configuration updated successfully${NC}"
fi

# Step 5: Update run_chatty.sh
echo -e "\n${BLUE}Step 5: Updating run_chatty.sh to use the new model...${NC}"
if confirm "Update the startup script to use Phi-4-small?"; then
    echo "Backing up the original run_chatty.sh..."
    cp /home/ubuntu/ai_in_a_box/run_chatty.sh /home/ubuntu/ai_in_a_box/run_chatty.sh.bak
    
    echo "Updating the model parameter..."
    sed -i 's/orca3b-4bit/phi4-small/g' /home/ubuntu/ai_in_a_box/run_chatty.sh
    
    echo -e "${GREEN}Startup script updated successfully${NC}"
fi

# Step.6: Check if service is running and restart
echo -e "\n${BLUE}Step 6: Checking if service is running...${NC}"
SERVICE_RUNNING=$(systemctl is-active run-chatty-startup)

if [ "$SERVICE_RUNNING" = "active" ]; then
    echo "The AI in a Box service is currently running"
    if confirm "Would you like to restart the service to apply changes?"; then
        echo "Restarting service..."
        systemctl restart run-chatty-startup
        echo -e "${GREEN}Service restarted successfully${NC}"
    else
        echo -e "${YELLOW}Service not restarted. Changes will apply on next reboot.${NC}"
    fi
else
    echo -e "${YELLOW}Service is not currently running${NC}"
    if confirm "Would you like to start the service now?"; then
        echo "Starting service..."
        systemctl start run-chatty-startup
        echo -e "${GREEN}Service started successfully${NC}"
    fi
fi

# Step 7: Clean up
echo -e "\n${BLUE}Step 7: Cleaning up...${NC}"
if [ -d "/tmp/phi4-download" ]; then
    rm -rf /tmp/phi4-download
fi

if [ -n "$USB_DRIVE" ]; then
    echo "Unmounting USB drive..."
    umount /media/usb
    echo -e "${GREEN}USB drive unmounted successfully${NC}"
fi

# Final message
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Upgrade to Phi-4 completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "The AI in a Box should now be running with the Phi-4-small model."
echo -e "Original files were backed up to: ${YELLOW}$BACKUP_DIR${NC}"
echo -e "If you experience any issues, you can restore from the backup files."
echo -e "\nEnjoy your upgraded AI in a Box! 🤖✨"
