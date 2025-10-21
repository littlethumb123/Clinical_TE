import logging
import google.cloud.logging
from google.auth import impersonated_credentials
from enum import Enum

# This module is tested with google-cloud-logging==3.1.2

# https://protect-us.mimecast.com/s/rLq-CrknY4F2AA6z4h4xPhU?domain=cloud.google.com

class CustomLogging:
    def __init__(self, workflow, task_id, execution_date, impersonation_service_account=None,):
        # Setting up the variables
        self.workflow=workflow
        self.task_id=task_id
        self.execution_date=execution_date
        #self.log_level=log_level
        self.labels={
                "workflow": self.workflow,
                "task-id": self.task_id,
                "execution-date": self.execution_date
            }
 
        # Setting up credentials and client for logging 
        if impersonation_service_account is None:
            logging_client = google.cloud.logging.Client()
        else:
            credentials, project=google.auth.default()
            target_credential=impersonated_credentials.Credentials(credentials, target_principal=impersonation_service_account, target_scopes=["https://protect-us.mimecast.com/s/7HlsCv2rj4CAWW4PrTAaz4M?domain=googleapis.com"])
            logging_client = google.cloud.logging.Client(credentials=target_credential)
 
        # Client Setup - standard python logger interface
        self.logger = logging_client.logger(name="airflow-worker")
        #, labels=self.labels)

    def __severity_check(self, severity):
        if severity in ["DEFAULT", "DEBUG", "INFO", "NOTICE", "WARNING", "ERROR", "CRITICAL", "EMERGENCY", "ALERT"]:
            return True
        else:
            return False

    def logJson(self, msg_dict, severity="INFO"):
        if self.__severity_check(severity):
            self.logger.log_struct(info=msg_dict, labels=self.labels, severity=severity)
        else:
            self.logger.log_struct(info=msg_dict, labels=self.labels, severity="INFO")
 
    def logText(self, msg_text, severity="INFO"):
        if self.__severity_check(severity):
            self.logger.log_text(text=msg_text, labels=self.labels, severity=severity)
        else:
            self.logger.log_text(text=msg_text, labels=self.labels, severity="INFO")


# This class uses cloud logging handler to tie the default python logger with cloud logging
class CustomLoggingWithHandler:
    def __init__(self, workflow, task_id, execution_date, log_level=logging.INFO, impersonation_service_account=None):
        # Setting up the variables
        self.workflow=workflow
        self.task_id=task_id
        self.execution_date=execution_date
        self.log_level=log_level
        self.labels={
                "workflow": self.workflow,
                "task-id": self.task_id,
                "execution-date": self.execution_date
            }
 
        # Setting up credentials and client for logging 
        if impersonation_service_account is None:
            logging_client = google.cloud.logging.Client()
        else:
            credentials, project=google.auth.default()
            target_credential=impersonated_credentials.Credentials(credentials, target_principal=impersonation_service_account, target_scopes=["https://protect-us.mimecast.com/s/7HlsCv2rj4CAWW4PrTAaz4M?domain=googleapis.com"])
            logging_client = google.cloud.logging.Client(credentials=target_credential)
 
        # Client Setup - standard python logger interface
        # https://protect-us.mimecast.com/s/Vpt2Cwpvk4fyLLP4gT1fYGh?domain=docs.python.org
        handler=google.cloud.logging.handlers.CloudLoggingHandler(logging_client, name="airflow-worker", labels=self.labels)
        google.cloud.logging.handlers.setup_logging(handler)
        logging.getLogger().setLevel(self.log_level)

    def logJson(self, msg_dict, severity=logging.INFO):
        if severity==logging.DEBUG:
            logging.debug(msg_dict)
        elif severity==logging.INFO:
            logging.info(msg_dict)
        elif severity==logging.WARN:
            logging.warn(msg_dict)
        elif severity==logging.ERROR:
            logging.error(msg_dict)
        elif severity==logging.CRITICAL:
            logging.critical(msg_dict)
 
    def logText(self, msg_text, severity=logging.INFO):
        if severity==logging.DEBUG:
            logging.debug(msg_text)
        elif severity==logging.INFO:
            logging.info(msg_text)
        elif severity==logging.WARN:
            logging.warn(msg_text)
        elif severity==logging.ERROR:
            logging.error(msg_text)
        elif severity==logging.CRITICAL:
            logging.critical(msg_text)