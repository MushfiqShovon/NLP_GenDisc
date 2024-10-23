import csv

filename = open('train_content.txt.token', 'r')
csv_reader = csv.reader(open('train.csv', 'r'), delimiter=',')
out_f = open('train.data', 'w')
for line, row in zip(filename, csv_reader):
        out_f.write(row[0] + '\t' + line.strip() + '\n')

filename = open('valid_content.txt.token', 'r')
csv_reader = csv.reader(open('valid.csv', 'r'), delimiter=',')
out_f = open('valid.data', 'w')
for line, row in zip(filename, csv_reader):
        out_f.write(row[0] + '\t' + line.strip() + '\n')

filename = open('test_content.txt.token', 'r')
csv_reader = csv.reader(open('test.csv', 'r'), delimiter=',')
out_f = open('test.data', 'w')
for line, row in zip(filename, csv_reader):
        out_f.write(row[0] + '\t' + line.strip() + '\n')