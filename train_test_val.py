import argparse 

def train_test_val():
    i = 0
    for line1, line2 in zip(tokens_file, alignments_file):

        line1 = line1.strip()
        line2 = line2.strip()

        if i < 1024:
            test_tokens_file.write(line1+"\n")
            test_alignments_file.write(line2+"\n")
        elif i > 1023 and i <=2047:
            val_tokens_file.write(line1+"\n")
            val_alignments_file.write(line2+"\n")
        else:
            train_tokens_file.write(line1+"\n")
            train_alignments_file.write(line2+"\n")
        i = i + 1
    return 0

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Traingi the bilstm for sentiment classification')
    parser.add_argument('--tokens_file', type = str, help = 'path to the  file')
    parser.add_argument('--alignment_file', type = str, help = 'path to the alignment file')
    parser.add_argument('--language', type = str, help = 'language')
    args = parser.parse_args()

    tokens_file = open(args.tokens_file, 'r')
    alignments_file = open(args.alignment_file, 'r')

    train_tokens_file_name = 'train/'+args.language+'-en.tokens'
    train_alignments_file_name = 'train/'+args.language+'-en.alignments'
    test_tokens_file_name = 'test/'+args.language+'-en.tokens'
    test_alignments_file_name = 'test/'+args.language+'-en.alignments'
    val_tokens_file_name = 'val/'+args.language+'-en.tokens'
    val_alignments_file_name = 'val/'+args.language+'-en.alignments'

    train_tokens_file = open(train_tokens_file_name, 'w')
    train_alignments_file = open(train_alignments_file_name, 'w')
    test_tokens_file = open(test_tokens_file_name, 'w')
    test_alignments_file = open(test_alignments_file_name, 'w')
    val_tokens_file = open(val_tokens_file_name, 'w')
    val_alignments_file = open(val_alignments_file_name, 'w')
    

    train_test_val()

    
    