#!/bin/bash
#
# USRP Gain Control Helper Script
# Makes it easy to update gain during transmission
#

CONTROL_FILE="/tmp/usrp_gain.txt"

# Function to show usage
show_usage() {
    echo "🔧 USRP Gain Control Helper"
    echo "=============================="
    echo "Usage: $0 [GAIN_VALUE]"
    echo ""
    echo "Examples:"
    echo "  $0 70        # Set gain to 70 dB"
    echo "  $0 50        # Set gain to 50 dB"
    echo "  $0 status    # Show current gain"
    echo "  $0 help      # Show this help"
    echo ""
    echo "📝 Control file: $CONTROL_FILE"
    echo ""
}

# Function to check if control file exists
check_control_file() {
    if [ ! -f "$CONTROL_FILE" ]; then
        echo "❌ Control file not found: $CONTROL_FILE"
        echo "💡 Start a noise generator first to create the control file"
        exit 1
    fi
}

# Function to show current gain
show_current_gain() {
    check_control_file
    current_gain=$(cat "$CONTROL_FILE" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$current_gain" ]; then
        echo "📶 Current gain: $current_gain dB"
    else
        echo "❌ Could not read current gain"
        exit 1
    fi
}

# Function to set new gain
set_gain() {
    local new_gain=$1
    
    # Validate numeric input
    if ! [[ "$new_gain" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "❌ Invalid gain value: $new_gain"
        echo "💡 Gain must be a number (e.g., 70, 50.5)"
        exit 1
    fi
    
    # Validate gain range (typical USRP B200 range)
    if (( $(echo "$new_gain < 0" | bc -l) )) || (( $(echo "$new_gain > 89.8" | bc -l) )); then
        echo "⚠ Warning: Gain $new_gain dB may be outside typical range (0-89.8 dB)"
        read -p "Continue anyway? [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "🚫 Cancelled"
            exit 0
        fi
    fi
    
    # Show current gain first
    show_current_gain
    
    # Try to write new gain
    if echo "$new_gain" > "$CONTROL_FILE" 2>/dev/null; then
        echo "✅ Gain updated: $new_gain dB"
        echo "🔄 Change will be applied during next transmission cycle"
    else
        echo "❌ Permission denied. Trying with sudo..."
        if sudo bash -c "echo '$new_gain' > '$CONTROL_FILE'"; then
            echo "✅ Gain updated: $new_gain dB (used sudo)"
            echo "🔄 Change will be applied during next transmission cycle"
        else
            echo "❌ Failed to update gain even with sudo"
            echo "💡 Check file permissions: ls -la $CONTROL_FILE"
            exit 1
        fi
    fi
}

# Main script logic
case "$1" in
    "")
        show_usage
        exit 0
        ;;
    "help"|"-h"|"--help")
        show_usage
        exit 0
        ;;
    "status"|"current"|"show")
        show_current_gain
        exit 0
        ;;
    *)
        set_gain "$1"
        exit 0
        ;;
esac
