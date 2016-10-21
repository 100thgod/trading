# Set up hardcoded variables
DATA_DIR = '/Users/armtiger/Documents/Data'

# Get data
csvFiles = list(filter(isCsv, os.listdir(DATA_DIR)))
tickers = list(map(getTicker, csvFiles))

data = []
for i in range(len(tickers)):
    with open(fullfile(DATA_DIR, csvFiles[i]),'r') as csvFile:
        r = csv.reader(csvFile)
        lineCount = 1
        for row in r:
            if lineCount == 1:
                print('header line\n')
                fields = row
                data.append(dict(list(map(lambda x: (x, []), fields))))
            else:
                for j in range(len(fields)):
                    data[i][fields[j]].append(row[j])
            lineCount = lineCount + 1
data = alignData(data, tickers)