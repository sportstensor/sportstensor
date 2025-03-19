import wandb
import schedule
from datetime import datetime, timedelta
import time

# Set your wandb project and entity
PROJECT_NAME = 'sportstensor-vali-logs'
ENTITY_NAME = 'sportstensor'
DAYS_TO_KEEP = 2  # Number of days to keep logs and artifacts

# Authenticate with wandb
wandb.login()

# Initialize the API
api = wandb.Api()

# Calculate the cutoff date
cutoff_date = datetime.now() - timedelta(days=DAYS_TO_KEEP)

# Function to delete old runs
def delete_old_runs():
    # Get all runs in the project
    runs = api.runs(f"{ENTITY_NAME}/{PROJECT_NAME}")

    for run in runs:
        # Convert the run's created_at timestamp to a datetime object
        #run_created_at = datetime.strptime(run.created_at, '%Y-%m-%dT%H:%M:%S')
        run_created_at = datetime.strptime(run.created_at, '%Y-%m-%dT%H:%M:%SZ')

        # Check if the run is older than the cutoff date
        if run_created_at < cutoff_date:
            files = run.files()
            for file in files:
                print(f"Deleting file {file.name} in run {run.id}")
                file.delete()

            for artifact in run.logged_artifacts():
                print(f"Deleting artifact {artifact.id} in run {run.id}")
                artifact.delete()

            print(f"Deleting run {run.id} created at {run.created_at}")
            run.delete()

    print("\n")
    print("Old logs and artifacts cleanup completed.")
    print("\n--------------------------------------------------------------------")

if __name__ == "__main__":
    # Schedule the function to run every hour
    schedule.every(60).minutes.do(delete_old_runs)

    # Delete old runs
    delete_old_runs()

    # Delete old artifacts
    #delete_old_artifacts()

    # Run the scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(60)
