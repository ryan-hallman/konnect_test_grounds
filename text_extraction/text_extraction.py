import json
from datetime import datetime, timedelta
import os
import re

from nltk.corpus import words
import spacy
import en_core_web_sm


class ProcessText(object):
    """
    Responsible for the text processing once everything has been prepped.
    Typically will pass a single page's text contents to this class.

    manifest: dict
        Returns the structured data to be saved to json to be used in QA
    text_string: str
        passed from textract, this is what feeds the NLP engine
    sensitive_list: list
        List of sensitive items (HIV, Drug Alcohol Abuse, Mental Health) found
    is_cnr: boolean
        Whether or not this page was a certificate of no records
    request_classification: str
        Business logic to set the type of request
    name_similarity_threshold: float
        threshold for similarity of a name to be considered the same

    """
    def __init__(self, text_string):

        self.manifest = {}
        self.text_string = text_string
        self.entity_d = {'corporation_list': [], 'person_list': []}
        self.sensitive_list = []
        self.sensitive_codes = None
        self.is_cnr = False
        self.request_classification = 'Unknown'
        self.known_entities = []

        self.date_list = []
        self.sensitive_phrases = ['hiv', 'substance abuse', 'drug abuse', 'psychotherapy', 'psych']
        self.sensitive_icd10_codes = None
        self.nlp = en_core_web_sm.load()
        self.name_similarity_threshold = 0.9
        self.combined_words = words.words()
        mterms_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace(
            '/app/toolbox',''),'app/static/lists/medical_terminology.txt')
        with open(mterms_path,'r') as f:
            self.combined_words.extend([x.strip() for x in f.readlines()])

        icd10_sensitivity_codes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace(
            '/app/toolbox',''),'app/static/lists/icd10_sensitivity_codes.json')
        with open(icd10_sensitivity_codes_path,'r') as f:
            self.sensitive_icd10_codes = json.loads(f.read())

    def __len__(self):
        return len(self.manifest.keys())

    def __repr__(self):
        return 'text_extraction.ProcessText({})'.format(str(dict(self.manifest.items())))

    def get_all(self, patient_name=None):
        """
        @param patient_name: tuple of first name and last name
        """
        if not self.text_string:
            self.structure_data()
            return
        self.check_if_cnr()
        self.parse_dates(365)
        self.parse_sensitive_phrases()
        self.search_for_icd10_sensitivity_codes()
        self.get_entities(patient_name)
        self.structure_data()

    def structure_data(self):
        self.manifest = {'entity_digest': self.entity_d, 'sensitive_list':
            self.sensitive_list, 'date_list': self.date_list, 'raw_text': self.text_string}

    def check_if_cnr(self):
        has_no_records_words = 'no records' in self.text_string.lower()
        has_dmrs_words = 'dmrs' in self.text_string.lower()

        if has_no_records_words and has_dmrs_words:
            self.is_cnr = True

    def parse_dates(self, max_in_future=None):
        supported_formats = ['%m/%d/%Y', '%m/%d/%y', '%m-%d-%y', '%m-%d-%Y',
            '%m%d%Y', '%Y-%m-%d', '%Y%m%d','%b %d %Y', '%b %d, %Y', '%B %d, %Y',
            '%B %d %Y', '%B %-d, %Y', '%B %-d %Y', '%b %-d, %Y', '%b %-d %Y',
            '%d-%b-%Y', '%-d-%b-%Y', '%d-%B-%Y', '%-d-%B-%Y']
        r = None
        string_list = self.text_string.split()
        date_list = []
        for i_string in string_list:
            for sformat in supported_formats:
                try:
                    d = datetime.strptime(i_string, sformat)
                    if max_in_future:
                        f_cutoff = (datetime.utcnow() + timedelta(days=max_in_future))
                    else:
                        f_cutoff = datetime.strptime('2100-01-01', '%Y-%m-%d')

                    if datetime.strptime('1900', '%Y') < d < f_cutoff:
                        if d not in date_list:
                            date_list.append(d.strftime('%Y-%m-%d'))
                        break
                except:  # nosec
                    continue # nosec
        self.date_list = date_list

    def parse_sensitive_phrases(self):
        for phrase in self.sensitive_phrases:
            if phrase in self.text_string.replace('/',' ').lower().split():
                self.sensitive_list.append(phrase)

    def search_for_icd10_sensitivity_codes(self):
        """
        Uses a list of ICD10 codes that are considered sensitive
        @return:
        """
        self.sensitive_codes = set(self.sensitive_icd10_codes.keys()).intersection(tuple(self.text_string.replace('/',' ').split()))
        for icd10_code in self.sensitive_codes:
            self.sensitive_list.append('%s - %s' % (icd10_code, self.sensitive_icd10_codes[icd10_code]))

    def get_entities(self, patient_name):
        """
        :param patient_name: list
            takes a list of name and is used to see if it exists in the text of
            the document string. This is a backup method in case the name is
            not recognized by Spacy
        """
        text_string_modified = self.text_string.replace('/',' ')
        doc = self.nlp(text_string_modified)

        self.find_name_regex(patient_name)

        for entity in doc.ents:
            if entity.label_ == 'PERSON' and entity.text.lower() not in self.entity_d['person_list']:
                if self.entity_is_valid(entity):
                    self.entity_d['person_list'].append(entity.text.lower())
            elif entity.label_ == 'ORG' and entity.text.lower() not in self.entity_d['corporation_list']:
                if self.entity_is_valid(entity):
                    self.entity_d['corporation_list'].append(entity.text.lower())

    def find_name_regex(self, patient_name):
        if not patient_name:
            return
        first_name, last_name = patient_name
        results = re.findall("%s.*%s" % (first_name.lower(),last_name.lower()), self.text_string.lower())
        results.extend(re.findall("%s.*%s" % (last_name.lower(),first_name.lower()), self.text_string.lower()))
        for result in results:
            if len(result) > 70:
                # most likely grabbed too much
                continue
            if result not in self.entity_d['person_list']:
                self.entity_d['person_list'].append(result)

    def entity_is_valid(self, entity):
        """
        Attempts to reduce the non-sensical items Spacy will suggest as names
        :param entity: spacy entity object
        :return: bool
        """

        if self.has_digits(entity.text.lower()):
            return False

        if self.entity_is_dictionary_words(entity):
            return False

        return True

    def entity_is_dictionary_words(self, entity):
        """
        Tests to see if all words are in dictionary
        :param entity: spacy entity object
        :return:
        """
        words = entity.text.lower().split()
        word_list = []
        for word in words:
            word_list.append(word in self.combined_words)

        return all(word_list)

    @staticmethod
    def has_digits(string):
        """
        Ensure that none of the words contain a digit
        """
        for char in string:
            if char.isdigit():
                return True
        return False