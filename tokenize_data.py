import argparse 
from amseg.amharicNormalizer import AmharicNormalizer
from amseg.amharicSegmenter import AmharicSegmenter
import tkseem as tk
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize  
#indic_tokenize.trivial_tokenize(indic_string):
#remove_nuktas=False
#factory=IndicNormalizerFactory()
#normalizer=factory.get_normalizer("hi",remove_nuktas)
#output_text=normalizer.normalize(input_text)
#

def amharic():
    sent_punct = []
    word_punct = []
    segmenter = AmharicSegmenter(sent_punct,word_punct)
    for line in input_file:
        line = line.strip()
        words = segmenter.amharic_tokenizer(line)
        sentence = " ".join(words) + '\n'
        output_file.write(sentence)
    input_file.close()
    output_file.close()
    return 0

def malayalam():
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer(args.language)
    for line in input_file:
        line = line.strip()
        line=normalizer.normalize(line)
        words = indic_tokenize.trivial_tokenize(line)
        sentence = " ".join(words) + '\n'
        output_file.write(sentence)
    input_file.close()
    output_file.close()
    return 0

def sinhala():
    remove_nuktas=False
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer(args.language)
    for line in input_file:
        line = line.strip()
        line=normalizer.normalize(line)
        words = indic_tokenize.trivial_tokenize(line)
        sentence = " ".join(words) + '\n'
        output_file.write(sentence)
    input_file.close()
    output_file.close()
    return 0










if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Traingi the bilstm for sentiment classification')
    parser.add_argument('--input_file', type = str, help = 'path to data file')
    parser.add_argument('--output_file', type = str, help = 'path to the output file')
    parser.add_argument('--language', type = str, help = 'language')
    args = parser.parse_args()

    input_file = open(args.input_file, 'r')
    output_file = open(args.output_file, 'w')

    if 'ml' in args.input_file:
        malayalam()
    elif 'sd' in args.input_file:
        sindhi()
    elif 'am' in args.input_file:
        amharic()
    elif 'si' in args.input_file:
        sinhala()
    
    
    