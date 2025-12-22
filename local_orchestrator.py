import yaml
import subprocess
import sys
import logging
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("local_orchestrator")

def load_workflow(file_path: str) -> Dict[str, Any]:
    """Load the workflow definition from YAML."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load workflow file: {e}")
        sys.exit(1)

def execute_local_step(step_name: str, step_def: Dict[str, Any]):
    """
    Execute a single workflow step locally using Docker.
    """
    logger.info(f"➡️  Executing step: {step_name}")

    # Inspect the step definition
    # Structure often: { 'step_name': { 'call': '...', 'args': {...} } }
    # Or simplified logic list
    
    # We look for "call: run_cloud_run_job"
    call_type = step_def.get('call')
    
    # Load job image mapping
    try:
        with open("dockerfiles/job_image_mapping.yaml", "r") as f:
            mapping_data = yaml.safe_load(f)
            
        # Reverse map to find container for job
        # mapping_data has lists like ml_science_jobs: [job1, job2]
        # and image_mapping: { ml_science: "defi-pipeline-ml-science" }
        
        job_to_category = {}
        for category in ["web_scraping", "ml_science", "lightweight", "database"]:
            for job in mapping_data.get(f"{category}_jobs", []):
                job_to_category[job] = category
                
        # Mapping category to docker compose service name
        category_to_service = {
            "web_scraping": "runner-scraping",
            "ml_science": "runner-ml",
            "lightweight": "runner-lightweight",
            "database": "runner-database"
        }
            
    except Exception as e:
        logger.warning(f"⚠️  Could not load job mapping: {e}. Defaulting to runner-lightweight.")
        job_to_category = {}
        category_to_service = {}

    if call_type == 'run_cloud_run_job':
        args = step_def.get('args', {})
        step_key = args.get('step_key')
        
        if not step_key:
            logger.warning(f"⚠️  Step {step_name} is missing 'step_key' in args. Skipping.")
            return

        # Determine service
        category = job_to_category.get(step_key, "lightweight")
        service_name = category_to_service.get(category, "runner-lightweight")

        logger.info(f"   Mapped to script: {step_key} -> Service: {service_name}")
        
        # Construct Docker command
        cmd = [
            "docker", "compose", "exec", "-T", service_name, 
            "python", "pipeline_runner.py", step_key
        ]
        
        try:
            # Run command and stream output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True
            )
            
            # Stream output
            for line in process.stdout:
                print(f"| {line}", end='')
                
            process.wait()
            
            if process.returncode != 0:
                logger.error(f"❌ Step {step_name} failed with exit code {process.returncode}")
                # We stop the pipeline on failure
                sys.exit(process.returncode)
            else:
                logger.info(f"✅ Step {step_name} completed successfully.")

        except Exception as e:
            logger.error(f"Error executing docker command: {e}")
            sys.exit(1)

    elif 'parallel' in step_def:
        logger.info("🔀 Executing parallel branch (Sequential execution in local mode)")
        # For simplicity, we run parallel branches sequentially locally
        branches = step_def['parallel']['branches']
        for branch in branches:
            # Each branch is a list of steps usually named implicitly or explicitly
            # In existing yaml: - branch_name: { steps: [...] }
            # Wait, the structure in the file seems to be a list of maps
            for branch_name, branch_content in branch.items():
                logger.info(f"   Branch: {branch_name}")
                if 'steps' in branch_content:
                   run_steps(branch_content['steps'])
        
    else:
        logger.info(f"ℹ️  Skipping non-job step type: {call_type or 'Control Flow'}")


def run_steps(steps: List[Dict[str, Any]]):
    """
    Iterate through a list of steps and execute them.
    """
    for step in steps:
        # Step is a dict with a single key usually: { 'step_name': { ... } }
        for step_name, step_content in step.items():
            execute_local_step(step_name, step_content)
            # Simple handling of 'next' is implicit in the list order for now
            # Complex jumps are not supported in this simple translator

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local Workflow Orchestrator")
    parser.add_argument("--config", default="workflow.yaml", help="Path to workflow.yaml")
    args = parser.parse_args()

    workflow = load_workflow(args.config)
    
    # Entry point commonly 'main'
    main_workflow = workflow.get('main', {})
    steps = main_workflow.get('steps', [])
    
    if not steps:
        logger.error("No steps found in 'main' workflow.")
        sys.exit(1)
        
    logger.info("🚀 Starting Local Pipeline Execution")
    run_steps(steps)
    logger.info("🎉 Local Pipeline Execution Completed")

if __name__ == "__main__":
    main()
