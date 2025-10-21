import json
import logging
import os
from airflow.utils.email import send_email
os.system('pip3 install pendulum')
import pendulum


PROJECT_ID = os.environ.get("GCP_PROJECT")
ENV = PROJECT_ID.split('-')[-1]

def merge(*args):
    j = {}
    for i in args:
        j.update(i)
    return j


date_timezone=str(pendulum.now('US/Eastern'))

def email_function(context,SUB_APP,status,to_email):
    dag_run = context.get('dag_run')
    if dag_run and dag_run.conf["recipient"]:
        to_email = dag_run.conf["recipient"]
    task_id = context["task_instance"].task_id
    DAG_ID = context["task_instance"].dag_id
    logs_url = context.get('task_instance').log_url
    if status == "Success":
        msg = '''
            <tr>
                    <p>Hi ,</p> 
                    <p>Below are the details about the run</p>
                    
                    <p>Dag_id   : %s,</p> 
                    <p>Task_id  : %s,</p> 
                    <p>Status   : <b style="color:green;">%s</b>,</p>
                    <p>Date_Time: %s,</p>
                    <p>Log_URL  : %s,</p>
                    <p>              </p>
                    <p>Thanks,\n     </p>
                    <p>IO-OPS       </p>
            </tr>
          '''% (DAG_ID,task_id,status,date_timezone,logs_url) 
    else:
        msg = '''
            <tr>
                    <p>Hi ,</p> 
                    <p>Below are the details about the run</p>
                    
                    <p>Dag_id   : %s,</p> 
                    <p>Task_id  : %s,</p> 
                    <p>Status   :  <b style="color:red;">%s</b>,</p>
                    <p>Date_Time: %s,</p>
                    <p>Log_URL  : %s,</p>
                    <p>              </p>
                    <p>Thanks,\n     </p>
                    <p>IO-OPS       </p>
            </tr>
          '''% (DAG_ID,task_id,status,date_timezone,logs_url)
        
        
    subject = SUB_APP+ " : Composer Dag Status: task id: " + task_id + "   " + status + "!!"  

    send_email(to=to_email, subject=subject, html_content=msg)