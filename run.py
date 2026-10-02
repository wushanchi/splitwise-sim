import logging
import json
import os
import random
import sys

import hydra

from hydra.utils import instantiate
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from simulator import TraceSimulator
from initialize import *


# register custom hydra resolver
OmegaConf.register_new_resolver("eval", eval)


def run_simulation(cfg):
    hardware_repo = init_hardware_repo(cfg)
    model_repo = init_model_repo(cfg)
    orchestrator_repo = init_orchestrator_repo(cfg)
    performance_model = init_performance_model(cfg)
    power_model = init_power_model(cfg)
    cluster = init_cluster(cfg)
    router = init_router(cfg, cluster)
    arbiter = init_arbiter(cfg, cluster)
    applications = init_applications(cfg, cluster, router, arbiter)
    trace = init_trace(cfg)
    for application in applications.values():
        router.add_application(application)
        arbiter.add_application(application)
    sim = TraceSimulator(trace=trace,
                         cluster=cluster,
                         applications=applications,
                         router=router,
                         arbiter=arbiter,
                         end_time=cfg.end_time)
    init_start_state(cfg,
                     cluster=cluster,
                     applications=applications,
                     router=router,
                     arbiter=arbiter)
    simulation_succeeded = False
    try:
        sim.run()
        simulation_succeeded = True
    finally:
        save_simulation_metadata(cfg, sim, trace, router, applications,
                                 performance_model, power_model,
                                 simulation_succeeded)


def save_simulation_metadata(cfg, sim, trace, router, applications,
                             performance_model, power_model, succeeded):
    """Keep profile provenance and completion counts even when execution fails."""
    if hasattr(performance_model, "get_metadata"):
        completed = len(router.completed_queue)
        metadata = {
            "execution": "cpu_discrete_event_simulation",
            "run_status": "completed" if succeeded else "failed",
            "num_input_requests": trace.num_requests,
            "num_completed_requests": completed,
            "num_uncompleted_requests": trace.num_requests - completed,
            "all_requests_completed": completed == trace.num_requests,
            "simulated_time_seconds": float(sim.time),
            "end_time_seconds": cfg.end_time,
            "performance_model": performance_model.get_metadata(),
            "power_model": (power_model.get_metadata()
                            if hasattr(power_model, "get_metadata")
                            else {"class": type(power_model).__name__,
                                  "dynamic_energy_validated": False}),
            "schedulers": {
                str(app_id): (app.scheduler.get_metadata()
                              if hasattr(app.scheduler, "get_metadata")
                              else {"class": type(app.scheduler).__name__})
                for app_id, app in applications.items()
            },
        }
        with open("simulation_metadata.json", "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    # print config
    #print(OmegaConf.to_yaml(cfg, resolve=False))
    #hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #print(OmegaConf.to_yaml(hydra_cfg, resolve=False))

    # initialize random number generator
    random.seed(cfg.seed)

    # delete existing oom.csv if any
    if os.path.exists("oom.csv"):
        os.remove("oom.csv")

    run_simulation(cfg)


if __name__ == "__main__":
    run()
