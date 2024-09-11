# -*- coding: utf-8 -*-
from task.ifind_market_state import get_market_state
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule


deployment = Deployment.build_from_flow(
    flow=get_market_state,
    name="market-state-flow",
    work_pool_name="ifind-agent-pool",
    schedule=(CronSchedule(cron="5 15 * * *", timezone="Asia/Shanghai")),
)

deployment.apply()

# import asyncio
#
# def main():
#     deployment = Deployment.build_from_flow(
#         flow=get_market_state,
#         name="market-state-flow",
#         work_pool_name="ifind-agent-pool",
#         schedule=(CronSchedule(cron="5 15 * * *", timezone="Asia/Shanghai")),
#     )
#     deployment.apply()
#
# asyncio.run(main())