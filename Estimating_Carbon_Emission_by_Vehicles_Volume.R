# Code Cell
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error

# Code Cell
cd /content/Estimating-Carbon-Emission-by-Vehicles-Volume-using-DataAnalysis-and-Machine-learning/

# Code Cell
df = pd.read_csv('yarra-traffic-counts.csv')
df.head()

# Code Cell
df.info()

# Code Cell
df.describe()

# Code Cell
print('Road_Name:\n')
df['road_name'].value_counts()[:10].to_frame().style.bar()

# Code Cell
print('Section Start:\n')
df.section_start.value_counts()[:10].to_frame().style.bar()

# Code Cell
print('Section End:\n')
df.section_end.value_counts()[:10].to_frame().style.bar()

# Code Cell
print('Suburb:\n')
df.suburb.value_counts()[:5].to_frame().style.bar()

# Code Cell
df_volume_pd_sum = df.groupby(["road_name",'section_start','section_end','suburb']).volume_per_day.sum().reset_index()
df_volume_pd_sum = df_volume_pd_sum.sort_values(by=['volume_per_day'],ascending=False)
df_volume_pd_sum.head()

# Code Cell
df_volume_pd_sum = df_volume_pd_sum[:10]
df_volume_pd_sum.plot(kind='barh')
plt.title('10 top Number of locations record where volume of vehicles are high')
plt.xlabel('Volume')
plt.ylabel('Rows Index')

# Code Cell
correlation = df.corr()
correlation

# Code Cell
# Visualization of correlation matrix heatmap 
plt.figure(figsize=(5,5))
sns.heatmap(correlation, annot = True)
plt.show()

# Code Cell
# Independent variables
x = df.iloc[:, [6,8,10]].values

# Dependent variable
y = df.iloc[:, 9].values

# Code Cell
print('Shape of independent variables:\n',x.shape,'\n','-' * 80)
print('Shape of dependent variable:\n',y.shape)

# Code Cell
y=y.reshape(-1,1)

# Code Cell
y.shape

# Code Cell
# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x = sc_X.fit_transform(x)
y = sc_y.fit_transform(y)

# Code Cell
# Choose the model
SVR_model= SVR()

# Code Cell
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.40, random_state=42)

# Code Cell
# Train the model
SVR_model.fit(x_train,y_train)

# Code Cell
y_pred = SVR_model.predict(x_test)
y_pred[:15]

# Code Cell
y_test[:15]

# Code Cell
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse