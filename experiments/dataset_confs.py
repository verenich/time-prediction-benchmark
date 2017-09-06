
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
for formula in range(1,5):
    dataset = "bpic2011_f%s"%formula 
    
    filename[dataset] = "labeled_logs_csv_processed/BPIC11_f%s.csv"%formula
    
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
    for formula in range(2,3):
        dataset = "bpic2015_%s_f%s"%(municipality, formula)
        
        filename[dataset] = "labeled_logs_csv_processed/BPIC15_%s_f%s.csv"%(municipality, formula)

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

filename[dataset] = "labeled_logs_csv_processed/traffic_fines_f3.csv"

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