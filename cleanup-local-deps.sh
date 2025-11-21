#!/bin/bash
# Script to clean up local Python dependencies (use with caution!)

set -e

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${YELLOW}⚠️  WARNING: This will uninstall Python packages from your system!${NC}"
echo -e "${YELLOW}Make sure you're using Docker for the project before proceeding.${NC}"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo -e "${GREEN}Cleaning up local dependencies...${NC}"

# List of packages to potentially remove (be careful!)
PACKAGES=(
    "jax"
    "jaxlib"
    "equinox"
    "thrml"
    "geoopt"
)

# Check what's installed
echo -e "${YELLOW}Checking installed packages...${NC}"
for pkg in "${PACKAGES[@]}"; do
    if pip3 list | grep -q "^$pkg "; then
        echo "  Found: $pkg"
        read -p "  Remove $pkg? (y/n): " remove
        if [ "$remove" = "y" ]; then
            pip3 uninstall -y "$pkg" 2>/dev/null || echo "    Could not remove $pkg"
        fi
    fi
done

echo -e "${GREEN}Cleanup complete!${NC}"
echo -e "${GREEN}Now use Docker for all development: ./docker-helper.sh build${NC}"

