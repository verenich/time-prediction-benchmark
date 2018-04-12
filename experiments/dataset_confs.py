
case_id_col = {}
activity_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

#### BPIC2011 settings ####
dataset = "bpic2011"

filename[dataset] = "labeled_logs_csv_processed/BPIC11.csv"

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity code"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "deviant"
neg_label[dataset] = "regular"

# features for classifier
dynamic_cat_cols[dataset] = ["Activity code", "Producer code", "Section", "Specialism code", "group"]
static_cat_cols[dataset] = ["Diagnosis", "Treatment code", "Diagnosis code", "case Specialism code", "Diagnosis Treatment Combination ID"]
dynamic_num_cols[dataset] = ["Number of executions", "duration", "month", "weekday", "hour"]
static_num_cols[dataset] = ["Age"]
    

    
#### BPIC2015 settings ####
for municipality in range(1,6):

    dataset = "bpic2015_%s"%municipality

    filename[dataset] = "labeled_logs_csv_processed/BPIC15_%s.csv"%municipality

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "remtime"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", "monitoringResource", "question", "Resource"]
    static_cat_cols[dataset] = ["Responsible_actor"]
    dynamic_num_cols[dataset] = ["duration", "month", "weekday", "hour"]
    static_num_cols[dataset] = ["SUMleges", 'Aanleg (Uitvoeren werk of werkzaamheid)', 'Bouw', 'Brandveilig gebruik (vergunning)', 'Gebiedsbescherming', 'Handelen in strijd met regels RO', 'Inrit/Uitweg', 'Kap', 'Milieu (neutraal wijziging)', 'Milieu (omgevingsvergunning beperkte milieutoets)', 'Milieu (vergunning)', 'Monument', 'Reclame', 'Sloop']

    if municipality in [3,5]:
        static_num_cols[dataset].append('Flora en Fauna')
    if municipality in [1,2,3,5]:
        static_num_cols[dataset].append('Brandveilig gebruik (melding)')
        static_num_cols[dataset].append('Milieu (melding)')



#### BPIC2017 settings ####
dataset = "bpic2017"

filename[dataset] = "labeled_logs_csv_processed/BPIC17.csv"

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
neg_label[dataset] = "regular"
pos_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ["Activity", 'Resource', 'Action', 'CreditScore', 'EventOrigin', 'lifecycle:transition'] 
static_cat_cols[dataset] = ['ApplicationType', 'LoanGoal']
dynamic_num_cols[dataset] = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', "duration", "month", "weekday", "hour", "activity_duration"]
static_num_cols[dataset] = ['RequestedAmount']



#### Traffic fines settings ####
dataset = "traffic_fines"

filename[dataset] = "labeled_logs_csv_processed/traffic_fines.csv"

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "deviant"
neg_label[dataset] = "regular"

# features for classifier
dynamic_cat_cols[dataset] = ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]
static_cat_cols[dataset] = ["article", "vehicleClass"]
dynamic_num_cols[dataset] = ["expense", "duration", "month", "weekday", "hour"]
static_num_cols[dataset] = ["amount", "points"]



#### Sepsis Cases settings ####
dataset = "sepsis"

filename[dataset] = "labeled_logs_csv_processed/Sepsis.csv"

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ["Activity", 'Diagnose', 'org:group']
static_cat_cols[dataset] = ['DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
       'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
       'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
       'SIRSCritTemperature', 'SIRSCriteria2OrMore']
dynamic_num_cols[dataset] = ['CRP', 'LacticAcid', 'Leucocytes', "duration", "month", "weekday", "hour"]
static_num_cols[dataset] = ['Age']



#### BPI2012A settings ####
dataset = "bpic2012a"

filename[dataset] = "labeled_logs_csv_processed/BPI2012A.csv"

case_id_col[dataset] = "case_id"
activity_col[dataset] = "activity_name"
timestamp_col[dataset] = "time"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ['activity_name', 'Resource']
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['open_cases','elapsed']
static_num_cols[dataset] = ['AMOUNT_REQ']



#### BPI2012O settings ####
dataset = "bpic2012o"

filename[dataset] = "labeled_logs_csv_processed/BPI2012O.csv"

case_id_col[dataset] = "case_id"
activity_col[dataset] = "activity_name"
timestamp_col[dataset] = "time"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ['activity_name', 'Resource']
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['open_cases','elapsed']
static_num_cols[dataset] = ['AMOUNT_REQ']



#### BPI2012W settings ####
dataset = "bpic2012w"

filename[dataset] = "labeled_logs_csv_processed/BPI2012W.csv"

case_id_col[dataset] = "case_id"
activity_col[dataset] = "activity_name"
timestamp_col[dataset] = "time"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ['activity_name', 'Resource']
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['open_cases','elapsed','proctime']
static_num_cols[dataset] = ['AMOUNT_REQ']



#### Credit requirements settings ####
dataset = "credit"

filename[dataset] = "labeled_logs_csv_processed/CreditRequirement.csv"

case_id_col[dataset] = "case_id"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ['Activity','weekday','hour']
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['open_cases','timesincecasestart','timesincemidnight','activity_duration']
static_num_cols[dataset] = []



#### helpdesk settings ####
dataset = "helpdesk"

filename[dataset] = "labeled_logs_csv_processed/helpdesk.csv"

case_id_col[dataset] = "case_id"
activity_col[dataset] = "activity_name"
timestamp_col[dataset] = "time"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ['activity_name','Resource']
static_cat_cols[dataset] = ["customer", "product",   "responsible_section",  "seriousness",  "service_level",   "service_type",  "support_section"]
dynamic_num_cols[dataset] = ['open_cases','elapsed']
static_num_cols[dataset] = []


#### hospital billing settings ####
dataset = "hospital"

filename[dataset] = "labeled_logs_csv_processed/hospital_billing_977.csv"

case_id_col[dataset] = "case_id"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ["Activity", "Resource","actOrange","actRed", "blocked", "caseType", "diagnosis", "flagC","flagD", "msgCode", "msgType", "state", "version"]
static_cat_cols[dataset] = ["speciality"]
dynamic_num_cols[dataset] = ["msgCount",   "timesincelastevent",    "timesincecasestart",    "event_nr",    "weekday",    "hour",    "open_cases"]
static_num_cols[dataset] = []



#### minit invoice settings ####
dataset = "minit"

filename[dataset] = "labeled_logs_csv_processed/minit_invoice_10.csv"

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ["Activity", "Resource",  "ActivityFinalAction",  "EventType",  "weekday",  "hour"]
static_cat_cols[dataset] = ["CostCenter.Code",  "Supplier.City",    "Supplier.Name",    "Supplier.State"]
dynamic_num_cols[dataset] = ["open_cases",    "timesincelastevent",    "timesincecasestart",    "event_nr"]
static_num_cols[dataset] = ["InvoiceTotalAmountWithoutVAT"]
