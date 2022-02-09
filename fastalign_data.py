import argparse 

def fastalign_data_format():
    for line1, line2 in zip(src_file, tgt_file):
        line1 = line1.strip()
        line2 = line2.strip()
        sent1 = line1.split()
        sent2 = line2.split()
        sent1 = [word for word in sent1 if word != '�' and word != '\u200b']
        sent2 = [word for word in sent2 if word != '�' and word != '\u200b']
        sent1 = [word for word in sent1 if word != '�' and word != '\u200e']
        sent2 = [word for word in sent2 if word != '�' and word != '\u200e']
        line1 = " ".join(sent1)
        line2 = " ".join(sent2)
        sentence = line1+" "+"|||"+" "+line2 +"\n"
        output_file.write(sentence)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Traingi the bilstm for sentiment classification')
    parser.add_argument('--src_file', type = str, help = 'path to data file')
    parser.add_argument('--tgt_file', type = str, help = 'path to the output file')
    parser.add_argument('--output_file', type = str, help = 'language')
    args = parser.parse_args()

    src_file = open(args.src_file, 'r')
    tgt_file = open(args.tgt_file, 'r')
    output_file = open(args.output_file, 'w')

    fastalign_data_format()

    
    