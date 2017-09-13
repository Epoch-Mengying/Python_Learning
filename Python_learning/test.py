import csv
import urllib2

# Load the data
train_url = "https://datahack-prod.s3.ap-south-1.amazonaws.com/workshop_train_file/train_gbW7HTd.csv"
test_url = "https://datahack-prod.s3.ap-south-1.amazonaws.com/workshop_test_file/test_2AFBew7.csv"
train_resp = urllib2.urlopen(train_url)
train = csv.reader(train_resp)
test_resp = urllib2.urlopen(test_url)
test = csv.reader(test_resp)


# Univariate Analysis
