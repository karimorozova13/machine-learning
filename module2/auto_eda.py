from ydata_profiling import ProfileReport
import seaborn as sns

titanic = sns.load_dataset('titanic')
report = ProfileReport(titanic, title='Tutanic')
report.to_file('./derived/n01_titanic_report.html')