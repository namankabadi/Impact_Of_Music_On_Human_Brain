# in urls.py
from django.urls import path
from .views import index
from .views import contact
from . import views
from .views import upload_csv
from .views import insights

urlpatterns = [
    path('', index, name='index_html'),
    path('upload/', upload_csv, name='upload_csv'),
    path('contact.html', views.contact, name='contact'),
    path('about.html', views.about, name='about'),
    path('index.html', views.index, name='index_html'),
    path('results.html', views.results, name='result_html'),
    path('access_data.html', views.access_data, name='access_data'),
    path('system.html', views.system, name='system_html'),
    path('insights.html', views.insights, name='insights'),
    path('upload/about.html', views.about, name='about'),
    path('upload/index.html', views.index, name='index_html_page'),
    path('upload/contact.html', views.contact, name='upload_contact'),
    path('upload/system.html', views.system, name='system_upload'),
    path('upload/insights.html', views.insights, name='insights_upload'),
    path('upload_three_data/', views.upload_three_data, name='upload_three_data'),
    path('predict_form/', views.predict, name='predict'),
    path('predict/', views.predict, name='predict'),
]
