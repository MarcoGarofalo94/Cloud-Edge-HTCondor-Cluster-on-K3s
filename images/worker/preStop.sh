#!/bin/bash

# Notify central manager of this worker going down so that it can be removed from the pool
condor_off -peaceful

# Specify the username prefix
USERNAME_PREFIX="slot"

# Get the User IDs (UIDs) matching the username prefix
USER_IDS=$(grep "^$USERNAME_PREFIX" /etc/passwd | cut -d':' -f3)

# Loop through each user ID and find their processes
for USER_ID in $USER_IDS; do
    # Find PIDs of processes created by the user
    PIDS=$(ps -u $USER_ID -o pid=)

    if [ -n "$PIDS" ]; then
        echo "User with UID $USER_ID created processes with PIDs: $PIDS"

        # Wait for processes to finish
        while true; do
            all_processes_done=true
            for PID in $PIDS; do
                if ps -p $PID > /dev/null; then
                    all_processes_done=false
                    break
                fi
            done

            if $all_processes_done; then
                break
            fi

            sleep 5
        done

        echo "All processes for user with UID $USER_ID have finished."
    else
        echo "User with UID $USER_ID has no active processes."
    fi
done