## Insurance Fraud Detection

-- Objective
Help insurance companies reduce financial losses from fraudulent claims
Provide fraud risk scores and explanations to internal teams
Enable faster, data-driven auditing decisions



| **Month**                  | Month of the accident                                                                
| **WeekOfMonth**            | Week number within the month when the accident happened                          
| **DayOfWeek**              | Day of week of the accident                                                      
| **Make**                   | Car manufacturer (Honda, Ford)                                             
| **AccidentArea**           | Urban/Rural                                                                      
| **DayOfWeekClaimed**       | Day the claim was filed                                                          
| **MonthClaimed**           | Month the claim was filed                                                        
| **WeekOfMonthClaimed**     | Week of the month when claim was filed                                           
| **Sex**                    | Gender of the policyholder                                                       
| **MaritalStatus**          | Marital status of the policyholder                                               
| **Age**                    | Age of the policyholder                                                          
| **Fault**                  | Who was at fault — policyholder or third party                                   
| **PolicyType**             | Type of policy (Collision, Liability,etc)                                
| **VehicleCategory**        | Type of vehicle (Sedan, SUV, etc.)                                               
| **VehiclePrice**           | Price category of the vehicle                                                    
| **FraudFound_P**           |  1 = Fraud, 0 = Not Fraud   **Target variable**                              
| **PolicyNumber**           | Unique policy ID (not useful for prediction)                                           
| **RepNumber**              | Insurance agent or reporting officer ID                                                
| **Deductible**             | Fixed deductible amount (same for all? Then drop)                                     
| **DriverRating**           | 1–4 scale — likely driver's skill/risk                                           
| **Days_Policy_Accident**   | Time between policy start and accident (0-7, 8-15 days)                    
| **Days_Policy_Claim**      | Time between policy start and claim                                              
| **PastNumberOfClaims**     | Number of past claims by the same person                                         
| **AgeOfVehicle**           | Age range of the vehicle (0-1 year, 2-3 years)                                
| **AgeOfPolicyHolder**      | Age range of the policyholder (young, mid, senior)                               
| **PoliceReportFiled**      | Was a police report filed (Yes/No)                                               
| **WitnessPresent**         | Were witnesses present (Yes/No)                                                  
| **AgentType**              | Type of agent — Internal vs External                                             
| **NumberOfSuppliments**    | Number of supplemental reports or files                                          
| **AddressChange_Claim**   | How recently the person changed address before claim
| **NumberOfCars**           | Cars owned by the policyholder                                                   
| **Year**                   | Year of the incident —                                
| **BasePolicy**             | Policy base type (Liability, Collision,)                              

-- 15420 rows, 33 columns of data
-- Removed null values with mode
-- removed duplicates
-- removed outliers from Age
-- performed indepth EDA using matplotlib,seaborn

feature enginnering

-- Created is_new_customer feature for better understanding
-- created Days_Policy_Accident_num and Days_Policy_Claim_num PastNumberOfClaims_Num AgeOfVehicle_Num for perfect numerical value
-- creared claim_delay for cheking delay
-- creaated risk_score for cheking driving perfomance
-- created vehicle_age_group as new and old
-- created claim_delay_week for the weekwise report
-- 14871 rows, 40 columns of data

-- split whole column into 2 categorical and numerical columns
-- performed mutal infrormation,pointbiserialr, on the numerical value
-- encoded categorical columns using onehot and label encoding for feature selection
-- performed Chi-Square Test on the catogoical values, used selectkbest for statitical checking amoung the values
-- compined medium + high important features from 3 feature selection test
-- scaled data for model wised feature selection uisng standardscaler
-- data was highly imbalanced so used SMOTETomek to balance
-- used Counter for counting each fraud and non fruad
-- performed randomforest for feature selectin , Accuracy Score: 0.93, Only 33% of predicted frauds are actually fraud,Model finds only 2% of all real frauds , Very poor balance between precision & recall
-- random forest produced bad result so i used BalancedRandomForestClassifier and xgboost 
-- BalancedRandomForestClassifier has accuracy 0.94 but poorly fail to detect fruads
-- XGBoost performed best in fraud recall with .94 accuracy
-- used feature_importances_ collect all important features
-- compined these 4 test and listed final features
-- again cleaned and prepared new data freame with final listed featutre
-- trained 8 diffrent calssifcation models (knn,svc,decisiontree,Gaussian,randomfrst,gradianboosting,XGBClassifier)
-- Gradient Boosting has Best F1-score for fraud (0.25), good balance, svc has Highest recall (0.80), okay F1, naiva bays has Best recall (0.85), but low precision
-- found svc and naive best for Catch more frauds at some cost of false positive
-- Gradient Boosting best for Balance catching frauds & minimizing false alarms  with 87 accuracy
-- performed gridsearchvc and randomcv for both svc and gradianet boosting , improved f1 and and accuracy 6% more
-- saved model using pickle and build dynamic dashboard in stramlit for realtime prediciton

