#!/bin/bash

# Script by Elle Palmer
# Last edited 12/22/2023
# NOTES:
# Please save a copy of this script into your project directory as 000_run.sh - DO NOT MAKE CHANGES TO THE MASTER SCRIPT. 
# Change relevant variables to meet your project specs
# WORKING_DIRECT2 can be used to run edited version of the master scripts saved to your project file
# Date parameters must be fed as strings in YYYYMMDD format - ex. Dec 31, 2022 as 20221231
# Comment out or delete any scripts you are not interested in running for your analyses
# Run script from your home directory using sh <project_directory>.<000_run.sh>

#Failure options overview: https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425?permalink_comment_id=3945021
set -eu pipefail

# set today's date timestamp for use in logfile name
# Timezone variables in the US: 
    # Eastern: America/New_York
    # Central: America/Chicago
    # Mountain: America/Boise
    # Pacific:  America/Los_Angeles
export NOW=$(TZ=America/New_York date +"%m_%d_%Y_%H:%M:%S")

#WORKING_DIRECT automatically calls unedited master scripts from home directory. 
#WORKING DIRECT2 can be edited to call scripts you copied to project folder and changed.
#Set LOGFILE to the name you want the command line output saved to
export WORKING_DIRECT='Medicaid-and-Duals-GCP/standard_medicaid_code/sh_version'
export WORKING_DIRECT2='cacm-mdcd-new_member_model/dev_archive/OOT_pull'
export LOGFILE='logfile_'$NOW

#set to your project and writable BQ DB
export GCP_PROJECT='anbc-hcb-dev'
export GCP_DB='cm_medicaid_hcb_dev'

#Set to your email, cost center, and desired table storage length
export OWNER='palmere1_aetna_com'
export COST_CENTER=13070
export DEFAULT_EXP="INTERVAL 180 DAY"

#Set descriptive project prefix (all tables stored as noted in variable ST)
#Set dates for member data to be pulled from; dates are inclusive;
#Do not change ST unless you have a custom population ID table with the necessary columns: asdb_member_key, asdb_plan_key, and asdb_elig_dt.
export PREFIX='a534354_IP_2024_OOT'
export ELIG_START_DT='20231101'
export ELIG_END_DT='20231130'
export ST=$GCP_PROJECT'.'$GCP_DB'.'$PREFIX'_member_index'
export SDOH_YEAR='2023'

#send the output of console run to a log file. Comment out  line if you do not want the log file.
#if you want every command line output saved (ex. every second of the BQ run), use 2>&1 at the end
#if you just want final run succeed/fail messages, remove the 2>&1 
#other logging options described here: https://blog.tratif.com/2023/01/09/bash-tips-1-logging-in-shell-scripts/
exec >$WORKING_DIRECT2/$LOGFILE.txt 2>&1

# Script creates the membership table with monthly membership for the dates between ELIG_START_DT and ELIG_END_DT. Used as the base table for all subsequent scripts
#sh $WORKING_DIRECT2/001a_Membership_Index.sh

# Script creates the claim line table
sh $WORKING_DIRECT2/002_Med_Claims_yr1.sh
sh $WORKING_DIRECT2/002_Med_Claims_yr2.sh

# Script creates ED and IP case and summary tables, OP table, med_claim_flag, and summarized cost/utilization tables
# TODO: split into 4 files - ED, IP, OP, and med_claim_flag/cost summary
sh $WORKING_DIRECT2/003_Cost_and_Utilization_yr1.sh
sh $WORKING_DIRECT2/003_Cost_and_Utilization_yr2.sh

# Script pulls from the PPM conditions dictionary to make commonly used condition flags
#sh $WORKING_DIRECT2/004_Conditions.sh

# Script pulls Rx claim line and creates Rx summary table
#sh $WORKING_DIRECT2/006_Rx_Claims_yr1.sh
#sh $WORKING_DIRECT2/006_Rx_Claims_yr2.sh

# Script pulls basic demographic features
#sh $WORKING_DIRECT2/007_Demographics.sh

# Script creates the GeoID for use in collecting relevant ACS and BRFSS data
#sh $WORKING_DIRECT2/008_GeoID.sh

# Pulls census-tract level ACS data for members where Geo data is available
# summaries of ACS data created by CSDI team
#sh $WORKING_DIRECT2/009_ACS.sh

# Pulls census-tract level CSDI indices for members where Geo data is available
#sh $WORKING_DIRECT2/010_preventative.sh

# Pulls census-tract level CSDI indices for members where Geo data is available
#sh $WORKING_DIRECT2/011_CSDI_risk.sh

#Create embeddings pre-table
#sh $WORKING_DIRECT2/012_embeddings_creation.sh

# Join all prior engineered features into one easy to manage table
sh $WORKING_DIRECT2/013_non_embedding_feature_beast.sh

