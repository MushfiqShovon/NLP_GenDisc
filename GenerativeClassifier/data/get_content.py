import csv
import sys
csv.field_size_limit(sys.maxsize)

csv_reader = csv.reader(open('test.csv', 'r'), delimiter=',')
f_out = open('test_content.txt','w')
for row in csv_reader:
    f_out.write(row[2] + '\n')
f_out.close()

csv_reader = csv.reader(open('train.csv', 'r'), delimiter=',')
f_out = open('train_content.txt','w')
for row in csv_reader:
    f_out.write(row[2] + '\n')
f_out.close()

csv_reader = csv.reader(open('valid.csv', 'r'), delimiter=',')
f_out = open('valid_content.txt','w')
for row in csv_reader:
    f_out.write(row[2] + '\n')
f_out.close()
