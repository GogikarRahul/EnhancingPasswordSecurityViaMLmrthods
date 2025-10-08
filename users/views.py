from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.contrib import messages

import Enhancing_Password_Security_With_Supervised_Machine_Learning

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd



# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

# ============================== DATASET VIEW ============================================================


def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'data.csv'
    column_names = ['password','strength']
    df = pd.read_csv(path,header=None, delimiter="\t",names=column_names)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})


#================================ Training and Prediction ==================================================
 
def ml(request):

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split as tts
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier

    path = os.path.join(settings.MEDIA_ROOT, 'data.csv')
    df = pd.read_csv(path, error_bad_lines=False)  

    df = df.dropna(subset=['password']).reset_index(drop=True)
    df['password'] = df['password'].astype(str)

    def cal_len(x):
        x = str(x)
        return len(x)

    def cal_capL(x):
        x = str(x)
        cnt = sum(1 for i in x if i.isupper())
        return cnt

    def cal_smL(x):
        x = str(x)
        cnt = sum(1 for i in x if i.islower())
        return cnt

    def cal_spc(x):
        x = str(x)
        return len(x) - len(re.findall(r'[\w]', x))

    def cal_num(x):
        x = str(x)
        cnt = sum(1 for i in x if i.isnumeric())
        return cnt

    df['length'] = df['password'].apply(cal_len)
    df['capital'] = df['password'].apply(cal_capL)
    df['small'] = df['password'].apply(cal_smL)
    df['special'] = df['password'].apply(cal_spc)
    df['numeric'] = df['password'].apply(cal_num)

   
    img_path = os.path.join(settings.MEDIA_ROOT, "ml", "length.jpg")
    sns.countplot(x=df['length'], color='red')
    plt.title('Countplot for Length of Password')
    plt.xlabel('LENGTH')
    plt.ylabel('COUNT')
    plt.savefig(img_path)
    # plt.savefig(r'E:\python Django\naveen\Enhancing_Password_Security_With_Supervised_Machine_Learning\media\ml\length.jpg')
    plt.show()

 
    save_path = os.path.join(settings.MEDIA_ROOT, 'processed_data.csv')
    dataset = df.to_csv(save_path, index=False)

    y = df['strength'].values
    x = df[['length', 'capital', 'small', 'special', 'numeric']].values

    x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=300)
    }

    results = {}
    confusion_matrices = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': accuracy,
            'report': report
        }
        confusion_matrices[name] = confusion

        # save_loc = r'E:\python Django\naveen\Enhancing_Password_Security_With_Supervised_Machine_Learning\media\ml\\'
        save_loc =  os.path.join(settings.MEDIA_ROOT, "ml", " ")

        joblib.dump(model, f'{save_loc}{name.lower().replace(" ", "_")}_model.pkl')

    # Visualizations for each model's confusion matrix
    for name, confusion in confusion_matrices.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d')
        plt.title(f'Confusion Matrix - {name}')
        save_loc = os.path.join(settings.MEDIA_ROOT, "ml", " ")
        plt.savefig(f'{save_loc}{name.lower().replace(" ", "_")}_confusion.jpg')
        plt.show()

    return render(
        request,
        'users/ml.html',
        {
            'results': results,
            'dataset':dataset,
         }
    )


#============================== FEATURE ENGINEERING ==================================================

# Define feature engineering functions
def cal_len(x):
    x = str(x)
    return len(x)

def cal_capL(x):
    x = str(x)
    cnt = sum(1 for i in x if i.isupper())
    return cnt

def cal_smL(x):
    x = str(x)
    cnt = sum(1 for i in x if i.islower())
    return cnt

def cal_spc(x):
    x = str(x)
    return len(x) - len(re.findall(r'[\w]', x))

def cal_num(x):
    x = str(x)
    cnt = sum(1 for i in x if i.isnumeric())
    return cnt


# =================================== PREDICTION =============================================


from django.shortcuts import render
import numpy as np
import pandas as pd
import re
import joblib


def predict_strength(request):
    import os
    if request.method == 'POST':
        user_input = request.POST.get('password')


        user_df = pd.DataFrame({'password': [user_input]})
        
        user_df['password'] = user_df['password'].astype(str)
        user_df['length'] = user_df['password'].apply(cal_len)
        user_df['capital'] = user_df['password'].apply(cal_capL)
        user_df['small'] = user_df['password'].apply(cal_smL)
        user_df['special'] = user_df['password'].apply(cal_spc)
        user_df['numeric'] = user_df['password'].apply(cal_num)

        # Using the trained scaler and model for prediction
        scaler = joblib.load(os.path.join(settings.MEDIA_ROOT, 'scaler.pkl'))
        model = joblib.load(os.path.join(settings.MEDIA_ROOT, 'model.pkl'))

        # Scaling the user input data
        user_input_features = user_df[['length', 'capital', 'small', 'special', 'numeric']].values
        scaled_input = scaler.transform(user_input_features)

        # Making the prediction
        prediction = model.predict(scaled_input)

        return render(request, 'users/prediction.html',{'prediction':prediction})
    return render(request, 'users/predictForm.html',{})