#Loading Required Packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from  catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


#A function for easily LabelEncoding the columns of a dataframe
def label_encode_df(dataframe,cols):
    for col in cols:
        le=LabelEncoder()
        dataframe[col]=le.fit_transform(dataframe[col])
#Reading the train and test files
train=pd.read_csv("train_2.csv")
test=pd.read_csv("test_2.csv")

#Making folds for stacking
train["fold"]=0
i=1
for tr,ts in KFold(n_splits=5,shuffle=True,random_state=5).split(train):
    train.loc[list(ts),"fold"]=i
    i=i+1

#Combing train and test, and Label Encoding
comb=pd.concat([train,test],axis=0)
comb.loc[comb.trainee_engagement_rating.isna(),"trainee_engagement_rating"]=3
comb.reset_index(inplace=True,drop=True)
cat_encoded_features=['city_tier', 'difficulty_level', 'education', 'gender',
       'is_handicapped', 'program_id','trainee_id',"test_id",
       'program_type', 'test_type']
comb_encoded=comb.copy()
label_encode_df(comb_encoded,cat_encoded_features)

#Stacking different models (Catboost Models with different seeds)
stack=pd.DataFrame()
stack["id"]=train.id
stack["fold"]=train.fold
stack["is_pass"]=train.is_pass
model1={"model_name" : "model1", "n_estimators":2746,"model_vars" :['city_tier', 'difficulty_level', 'education', 'gender',
           'is_handicapped', 'program_id','trainee_id',
           'program_type', 'test_id', 'test_type', 'total_programs_enrolled',
           'trainee_engagement_rating',"age"],"cat_vars" :10,"seed" :3}
model2={"model_name" : "model2", "n_estimators":2746,"model_vars" :['city_tier', 'difficulty_level', 'education', 'gender',
           'is_handicapped', 'program_id','trainee_id',
           'program_type', 'test_id', 'test_type', 'total_programs_enrolled',
           'trainee_engagement_rating',"age"],"cat_vars" :10,"seed":12}
model3={"model_name" : "model3", "n_estimators":2746,"model_vars" :['city_tier', 'difficulty_level', 'education', 'gender',
           'is_handicapped', 'program_id','trainee_id',
           'program_type', 'test_id', 'test_type', 'total_programs_enrolled',
           'trainee_engagement_rating',"age"],"cat_vars" :10, "seed":30}
models=[model1,model2,model3]

#Running the models on 4 folds and predicting on the 5th
for model in models:
    stack[model["model_name"]]=0
    for i in range(1,6):
        print(model["model_name"])
        comb_encoded["dataset"]="train"
        len_train=73147
        comb_encoded.loc[73147:,"dataset"]="test"
        comb_encoded.loc[comb_encoded.fold==i,"dataset"]="valid"
        y=comb_encoded.loc[comb_encoded.dataset=="train","is_pass"].values
        y_test=comb_encoded.loc[comb_encoded.dataset=="valid","is_pass"].values
        x=comb_encoded.loc[comb_encoded.dataset=="train",model["model_vars"]].values
        x_test=comb_encoded.loc[comb_encoded.dataset=="valid",model["model_vars"]].values
        cat_model=CatBoostClassifier(eval_metric="AUC",n_estimators=model["n_estimators"],random_state=model["seed"])
        cat_model.fit(x,y,cat_features=list(range(0,model["cat_vars"])),verbose=False)
        stack.loc[stack.fold==i,model["model_name"]]=cat_model.predict_proba(comb_encoded.loc[comb_encoded.dataset=="valid",model["model_vars"]].values)[:,1]

#Running the base models and the whole train set and predicitng for test set
stack_test=pd.DataFrame()
stack_test["id"]=test.id
for model in models:
    stack_test[model["model_name"]]=0
    print(model["model_name"])
    comb_encoded["dataset"]="train"
    len_train=73147
    comb_encoded.loc[73147:,"dataset"]="test"
    #comb_encoded.loc[comb_encoded.fold==i,"dataset"]="valid"
    y=comb_encoded.loc[comb_encoded.dataset=="train","is_pass"].values
    #y_test=comb_encoded.loc[comb_encoded.dataset=="valid","is_pass"].values
    x=comb_encoded.loc[comb_encoded.dataset=="train",model["model_vars"]].values
    #x_test=comb_encoded.loc[comb_encoded.dataset=="valid",model["model_vars"]].values
    cat_model=CatBoostClassifier(eval_metric="AUC",n_estimators=model["n_estimators"],random_state=model["seed"])
    cat_model.fit(x,y,cat_features=list(range(0,model["cat_vars"])),verbose=False)
    stack_test[model["model_name"]]=cat_model.predict_proba(comb_encoded.loc[comb_encoded.dataset=="test",model["model_vars"]].values)[:,1]
    
#Running a stacked model on the 3 base models and making final predications
lr_model=LogisticRegression()
lr_model.fit(X=stack[["model1","model2","model3"]],y=stack.is_pass) 
stack_test["is_pass"]=lr_model.predict_proba(X=stack_test[["model1","model2","model3"]])[:,1]
stack_test[["id","is_pass"]].to_csv("sub_final.csv")
