import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error,confusion_matrix, accuracy_score,f1_score,classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


st.title('ML Application')

# Hàm để xử lý dữ liệu rỗng cho các cột được chọn
def handle_missing_values(df, columns, option):
    if option == "Xóa các giá trị missing":
        df = df.dropna(subset=columns)
    elif option == "Thay thế các giá trị missing":
        for col in columns:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    return df

# Hàm để chuyển đổi biến categorical thành số int64
def convert_categorical_to_numerical(df, columns):
    for column in columns:
        df[column] = df[column].astype('category').cat.codes.astype('int64')
    return df

# Hàm để lưu DataFrame mới
def save_new_dataframe(df):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Tải xuống DataFrame",
        data=csv,
        file_name='new_dataframe.csv',
        mime='text/csv',
    )

# 2. Importing datasets
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    st.subheader("Thông tin DataFrame")
    buffer = StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Chuyển đổi biến categorical thành số")
    cat_columns = data.select_dtypes(include=['object']).columns.tolist()
    columns_to_convert = st.multiselect("Chọn các cột để chuyển đổi", cat_columns)
    if columns_to_convert:
        data = convert_categorical_to_numerical(data, columns_to_convert)
        st.write(data)

    st.subheader("Chọn cột và xử lý dữ liệu rỗng")
    columns = data.columns.tolist()
    columns_to_handle = st.multiselect("Chọn các cột để xử lý dữ liệu rỗng", columns)
    if columns_to_handle:
        handle_option = st.selectbox("Chọn phương pháp xử lý", ["Xóa các giá trị missing", "Thay thế các giá trị missing"])
        data = handle_missing_values(data, columns_to_handle, handle_option)
        st.write(data)

    st.subheader("Chọn cột để xóa")
    columns_to_delete = st.multiselect("Chọn các cột để xóa", columns)
    if columns_to_delete:
        if st.button("Xác nhận xóa"):
            data = data.drop(columns_to_delete, axis=1)
            st.write("Đã xóa các cột được chọn")
            st.write(data)

    st.subheader("Kiểm tra và xóa dữ liệu trùng lặp")
    duplicates = data.duplicated().sum()
    st.write(f"Số lượng hàng bị trùng lặp: {duplicates}")
    if duplicates > 0:
        if st.button("Xóa các hàng trùng lặp"):
            data = data.drop_duplicates()
            st.write("Đã xóa các hàng trùng lặp")
            st.write(data)
                
    st.subheader("Lưu DataFrame mới")
    st.dataframe(data)
    save_new_dataframe(data)

    st.title("Visualling")
    st.subheader("Visualize Data")
    chart_type = st.selectbox("Chọn kiểu biểu đồ", ["Histogram", "Boxplot", "Scatterplot"])
    columns_to_plot = st.multiselect("Chọn cột để vẽ", data.columns)
    
    if st.button("Vẽ biểu đồ"):
        for column_to_plot in columns_to_plot:
            plt.figure(figsize=(10, 6))
            if chart_type == "Histogram":
                sns.histplot(data[column_to_plot], kde=True)
                plt.title(f"Histogram of {column_to_plot}")
            elif chart_type == "Boxplot":
                sns.boxplot(y=data[column_to_plot])
                plt.title(f"Boxplot of {column_to_plot}")
            elif chart_type == "Scatterplot":
                other_columns = data.columns.difference([column_to_plot]).tolist()
                x_axis = st.selectbox("Chọn cột x-axis", other_columns)
                sns.scatterplot(x=data[x_axis], y=data[column_to_plot])
                plt.title(f"Scatterplot of {column_to_plot} vs {x_axis}")
            st.pyplot(plt)
    

    st.title("Xây dựng mô hình học máy")
    model_option = st.selectbox("Chọn mô hình", ["Linear Regression", "Logistic Regression", "KNN", "Decision Tree", "Random Forest"])
    if model_option == "Linear Regression":
        # 3. Extracting Independent and Dependent Variable
        st.subheader('Extract Independent and Dependent Variables')
        X_columns = st.multiselect('Select Independent Variables (X)', data.columns)
        y_column = st.selectbox('Select Dependent Variable (y)', data.columns)

        if len(X_columns) == 0 or y_column is None:
            st.stop()

        X = data[X_columns].values
        y = data[y_column].values

        # 4. Feature Scaling
        st.subheader('Feature Scaling')
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        st.write(pd.DataFrame(X, columns=X_columns).head())

         # 5. Convert categorical variable to number
        st.subheader('Convert Categorical Variables to Numbers')
        if y.dtype == 'object':
            labelencoder_y = LabelEncoder()
            y = labelencoder_y.fit_transform(y)
        st.write(y[:5])

        st.write("Linear Regression model selected")
        # 6. Fitting Linear Regression to the training set
        st.subheader('Split the data and Fit Linear Regression Model')
        test_size = st.slider('Test size', 0.1, 0.5, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=43)
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        st.write('Model fitted successfully')

        # 7. Predicting the test set result
        st.subheader('Predict the Test Set Results')
        y_pred = regressor.predict(X_test)
        st.write(y_pred)
        plt.plot(y_pred)
        st.pyplot(plt)

        # 8. Evaluating the model
        st.subheader('Model Evaluation')
        mse = mean_squared_error(y_test, y_pred)
        st.write('Mean Squared Error: {:.2f}'.format(mse))

        # Visualization functions
        def plot_Linear_regression_2d(X, y, model, title, X_columns):
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, color='red', zorder=20)
            plt.plot(X, model.predict(X), color='blue', linewidth=3)
            plt.title(title)
            plt.xlabel(X_columns[0])
            plt.ylabel('Dependent Variable')
            st.pyplot(plt)

        def plot_3d(X, y, model, title, X_columns):
            if X.shape[1] != 2:
                st.error("Need exactly 2 features for 3D plot.")
                return
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis', edgecolor='k', s=50)
            x_surf, y_surf = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
            z_surf = model.predict(np.c_[x_surf.ravel(), y_surf.ravel()]).reshape(x_surf.shape)
            ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, rstride=100, cstride=100, color='blue', edgecolor='none')
            ax.set_title(title)
            ax.set_xlabel(X_columns[0])
            ax.set_ylabel(X_columns[1])
            ax.set_zlabel('Dependent Variable')
            st.pyplot(fig)

        # Select plot type
        plot_type = st.radio("Select plot type:", ('2D', '3D'))

        if plot_type == '2D':
            plot_Linear_regression_2d(X_test, y_test, regressor, 'Test Set (2D)', X_columns)
        else:
            plot_3d(X_test, y_test, regressor, 'Test Set (3D)', X_columns)

    elif model_option == "Logistic Regression":
        st.title("Logistic Regression model selected")
        # 3. Extracting Independent and Dependent Variable
        st.subheader('Extract Independent and Dependent Variables')
        X_columns = st.multiselect('Select Independent Variables (X)', data.columns)
        y_column = st.selectbox('Select Dependent Variable (y)', data.columns)

        if len(X_columns) == 0 or y_column is None:
            st.stop()

        X = data[X_columns].values
        y = data[y_column].values

        # 4. Feature Scaling
        st.subheader('Feature Scaling')
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        st.write(pd.DataFrame(X, columns=X_columns).head())

        # 5. Convert categorical variable to number
        st.subheader('Convert Categorical Variables to Numbers')
        if y.dtype == 'object':
            labelencoder_y = LabelEncoder()
            y = labelencoder_y.fit_transform(y)
        st.write(y[:5])

        # 6. Fitting Logistic Regression to the training set
        st.subheader('Split the data and Fit Logistic Regression Model')
        test_size = st.slider('Test size', 0.1, 0.5, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=43)
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)
        st.write('Model fitted successfully')

        # 7. Predicting the test set result
        st.subheader('Predict the Test Set Results')
        y_pred = classifier.predict(X_test)
        st.write(y_pred)

        # 8. Creating the Confusion matrix
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        st.pyplot(plt)

        # 9. Calculation accuracy
        st.subheader('Accuracy')
        accuracy = accuracy_score(y_test, y_pred)
        st.write('Accuracy: {:.2f}%'.format(accuracy * 100))

        st.subheader('Predict probabilities the Test Set Results')
        y_pred = classifier.predict_proba(X_test)
        st.write(y_pred)

        def plot_logistic_regression_2d(X, y, model, title, X_columns):
            plt.figure(figsize=(10, 6))
            
            # Plotting the points
            plt.scatter(X[:, 0], y, color='black', zorder=20)

            # Plotting the logistic regression curve
            def model_func(x):
                return 1 / (1 + np.exp(-x))
            
            X_test_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)
            loss = model_func(X_test_vals * model.coef_[0][0] + model.intercept_[0])
            plt.plot(X_test_vals, loss, color='blue', linewidth=3)
            
            plt.title(title)
            plt.xlabel(X_columns[0])
            plt.ylabel('Probability')
            st.pyplot(plt)

        def plot_3d(X, y, model, title, X_columns):
            if X.shape[1] < 3:
                st.error("Need at least 3 features for 3D plot.")
                return

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Scatter plot
            scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', edgecolor='k', s=50)
            
            # Create grid to plot decision boundary
            x = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
            y = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
            X1, X2 = np.meshgrid(x, y)
            X_grid = np.array([X1.ravel(), X2.ravel()]).T
            Z = -(model.intercept_ + model.coef_[0][0] * X1 + model.coef_[0][1] * X2) / model.coef_[0][2]
            Z = Z.reshape(X1.shape)
            
            # Plot decision boundary
            ax.plot_surface(X1, X2, Z, alpha=0.5, rstride=100, cstride=100, color='blue', edgecolor='none')
            
            ax.set_title(title)
            ax.set_xlabel(X_columns[0])
            ax.set_ylabel(X_columns[1])
            ax.set_zlabel(X_columns[2])
            st.pyplot(fig)

        # Select plot type
        plot_type = st.radio("Select plot type:", ('2D', '3D'))

        if plot_type == '2D':
            plot_logistic_regression_2d(X_train, y_train, classifier, 'Training Set (2D)', X_columns)
        else:
            plot_3d(X_train, y_train, classifier, 'Training Set (3D)', X_columns)

        # Visualizing the test set result 2d or 3d
        st.subheader('Visualizing Test Set Results')

        if plot_type == '2D':
            plot_logistic_regression_2d(X_test, y_test, classifier, 'Test Set (2D)', X_columns)
        else:
            plot_3d(X_test, y_test, classifier, 'Test Set (3D)', X_columns)



    elif model_option == "KNN":
        st.write("KNN model selected")
         
        # 3. Extracting Independent and Dependent Variable
        st.subheader('Extract Independent and Dependent Variables')
        X_columns = st.multiselect('Select Independent Variables (X)', data.columns)
        y_column = st.selectbox('Select Dependent Variable (y)', data.columns)

        if len(X_columns) == 0 or y_column is None:
            st.stop()

        X = data[X_columns].values
        y = data[y_column].values

        # 4. Feature Scaling
        st.subheader('Feature Scaling')
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        st.write(pd.DataFrame(X, columns=X_columns).head())

        # 5. Convert categorical variable to number
        st.subheader('Convert Categorical Variables to Numbers')
        if y.dtype == 'object':
            labelencoder_y = LabelEncoder()
            y = labelencoder_y.fit_transform(y)
        st.write(y[:5])

        # 6. Fitting KNN to the training set
        st.subheader('Split the data and Fit KNN Model')
        test_size = st.slider('Test size', 0.1, 0.5, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.subheader('Chọn số lượng (k)')
        k = st.slider('Chọn k', 1, 11, 5)

        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        st.write('Model fitted successfully')

        # 7. Predicting the test set result
        st.subheader('Predict the Test Set Results')
        y_pred = classifier.predict(X_test)
        st.write(y_pred)

        # 8. Creating the Confusion matrix
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        st.pyplot(plt)

        # f1 score
        f1score = f1_score(y_test, y_pred, average='weighted')
        st.write('F1 Score: {:.2f}%'.format(f1score * 100))

        # 9. Calculation accuracy
        st.subheader('Accuracy')
        accuracy = accuracy_score(y_test, y_pred)
        st.write('Accuracy: {:.2f}%'.format(accuracy * 100))

        # 10. Scatter Plot
        st.subheader('Scatter Plot of Actual vs Predicted')
        actual_values = y_test
        predicted_values = y_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted_values, cmap='coolwarm', edgecolor='k', alpha=0.7)
        plt.xlabel(X_columns[0])
        plt.ylabel(X_columns[1])
        
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Not Purchased', markersize=10, markerfacecolor='b'),
                            plt.Line2D([0], [0], marker='o', color='w', label='Purchased', markersize=10, markerfacecolor='r')],
                title="Actual vs Predicted")
        plt.title('Actual vs Predicted Purchased')
        st.pyplot(plt)
            
    elif model_option == "Decision Tree":
        st.write("Decision Tree model selected")
        # 3. Extracting Independent and Dependent Variable
        st.subheader('Extract Independent and Dependent Variables')
        X_columns = st.multiselect('Select Independent Variables (X)', data.columns)
        y_column = st.selectbox('Select Dependent Variable (y)', data.columns)

        if len(X_columns) == 0 or y_column is None:
            st.stop()

        X = data[X_columns].values
        y = data[y_column].values

        # 4. Feature Scaling
        st.subheader('Feature Scaling')
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        st.write(pd.DataFrame(X, columns=X_columns).head())

        # 5. Convert categorical variable to number
        st.subheader('Convert Categorical Variables to Numbers')
        if y.dtype == 'object':
            labelencoder_y = LabelEncoder()
            y = labelencoder_y.fit_transform(y)
        st.write(y[:5])

        # 6. Fitting DecisionTree to the training set
        st.subheader('Split the data and Fit DecisionTree Model')
        test_size = st.slider('Test size', 0.1, 0.5, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        classifier = DecisionTreeClassifier(random_state=0)
        classifier.fit(X_train, y_train)
        st.write('Model fitted successfully')

        # 7. Predicting the test set result
        st.subheader('Predict the Test Set Results')
        y_pred = classifier.predict(X_test)
        st.write(y_pred)

        # 8. Creating the Confusion matrix
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        st.pyplot(plt)

        # f1 score
        f1score = f1_score(y_test, y_pred, average='weighted')
        st.write('F1 Score: {:.2f}%'.format(f1score * 100))

        # 9. Calculation accuracy
        st.subheader('Accuracy')
        accuracy = accuracy_score(y_test, y_pred)
        st.write('Accuracy: {:.2f}%'.format(accuracy * 100))
    
        # 11. Visualizing the Decision Tree
        st.subheader('Decision Tree Visualization')
        fig = plt.figure(figsize=(20, 15))
        plot_tree(classifier, feature_names=X_columns, class_names=[str(c) for c in np.unique(y)], filled=True, fontsize=10)
        st.pyplot(fig)

                # 10. Classification Report
        st.subheader('Classification Report')
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)
            
    elif model_option == "Random Forest":
        st.write("Random Forest model selected")
        # 3. Extracting Independent and Dependent Variable
        st.subheader('Extract Independent and Dependent Variables')
        X_columns = st.multiselect('Select Independent Variables (X)', data.columns)
        y_column = st.selectbox('Select Dependent Variable (y)', data.columns)

        if len(X_columns) == 0 or y_column is None:
            st.stop()

        X = data[X_columns].values
        y = data[y_column].values

        # 4. Feature Scaling
        st.subheader('Feature Scaling')
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        st.write(pd.DataFrame(X, columns=X_columns).head())

        # 5. Convert categorical variable to number
        st.subheader('Convert Categorical Variables to Numbers')
        if y.dtype == 'object':
            labelencoder_y = LabelEncoder()
            y = labelencoder_y.fit_transform(y)
        st.write(y[:5])

        # 6. Fitting RandomForest to the training set
        st.subheader('Split the data and Fit RandomForest Model')
        test_size = st.slider('Test size', 0.1, 0.5, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        classifier = RandomForestClassifier(random_state=0)
        classifier.fit(X_train, y_train)
        st.write('Model fitted successfully')

        # 7. Predicting the test set result
        st.subheader('Predict the Test Set Results')
        y_pred = classifier.predict(X_test)
        st.write(y_pred)

        # 8. Creating the Confusion matrix
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        st.pyplot(plt)

        # f1 score
        f1score = f1_score(y_test, y_pred, average='weighted')
        st.write('F1 Score: {:.2f}%'.format(f1score * 100))

        # 9. Calculation accuracy
        st.subheader('Accuracy')
        accuracy = accuracy_score(y_test, y_pred)
        st.write('Accuracy: {:.2f}%'.format(accuracy * 100))

        # Feature Importance
        feature_importances = classifier.feature_importances_
        importance_df = pd.DataFrame({'Feature': X_columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=True)
        st.write(importance_df)
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightgreen')
        plt.xlabel('Importance')
        plt.title('Feature Importances')
        st.pyplot(plt)
        # 10. Classification Report
        st.subheader('Classification Report')
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)

else:
    st.write("Please upload a CSV file.")
