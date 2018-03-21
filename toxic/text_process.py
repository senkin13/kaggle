from __future__ import absolute_import
import re
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix


class text_processing(object):

    def __init__(self):
        # PREPROCESSING PART
        self.repl = {
            "&lt;3": " good ",
            ":d": " good ",
            ":dd": " good ",
            ":p": " good ",
            "8)": " good ",
            ":-)": " good ",
            ":)": " good ",
            ";)": " good ",
            "(-:": " good ",
            "(:": " good ",
            "yay!": " good ",
            "yay": " good ",
            "yaay": " good ",
            "yaaay": " good ",
            "yaaaay": " good ",
            "yaaaaay": " good ",
            ":/": " bad ",
            ":&gt;": " sad ",
            ":')": " sad ",
            ":-(": " bad ",
            ":(": " bad ",
            ":s": " bad ",
            ":-s": " bad ",
            "&lt;3": " heart ",
            ":d": " smile ",
            ":p": " smile ",
            ":dd": " smile ",
            "8)": " smile ",
            ":-)": " smile ",
            ":)": " smile ",
            ";)": " smile ",
            "(-:": " smile ",
            "(:": " smile ",
            ":/": " worry ",
            ":&gt;": " angry ",
            ":')": " sad ",
            ":-(": " sad ",
            ":(": " sad ",
            ":s": " sad ",
            ":-s": " sad ",
            r"\br\b": "are",
            r"\bu\b": "you",
            r"\bhaha\b": "ha",
            r"\bhahaha\b": "ha",
            r"\bdon't\b": "do not",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bhasn't\b": "has not",
            r"\bhaven't\b": "have not",
            r"\bhadn't\b": "had not",
            r"\bwon't\b": "will not",
            r"\bwouldn't\b": "would not",
            r"\bcan't\b": "can not",
            r"\bcannot\b": "can not",
            r"\bi'm\b": "i am",
            "m": "am",
            "r": "are",
            "u": "you",
            "haha": "ha",
            "hahaha": "ha",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "won't": "will not",
            "wouldn't": "would not",
            "can't": "can not",
            "cannot": "can not",
            "i'm": "i am",
            "m": "am",
            "i'll": "i will",
            "its": "it is",
            "it's": "it is",
            "'s": " is",
            "that's": "that is",
            "weren't": "were not",
            "fagget": "faggot",
            "conndoms": "condom",
            "condoms": "condom",
            "muthafucker": "motherfucker",
            "mothafucka": "motherfucker",
        }

        self.keys = [i for i in self.repl.keys()]

        # stopwords = {x: 1 for x in set(['the', 'i', 'of', 'to', ":", 'is', 'are', 'this', 'that', 'it', 'am', 'they'])}
        # stopwords = set(stop_file["stopwords"])
        self.stopwords = {x: 1 for x in set(
            ['the', 'i', 'of', 'to', ":", 'is', 'are', 'this', 'that', 'it', 'am', 'they',
             "talk", "contribution", "hope", "chat", "editing", "utc", "page", "contributions", "jun", "email",
             "maybe", "holla", "administrator", "wolfowitz", "snowman", "redirect", "user", "andemu", "administrators",
             "speak", "user_talk", "up", "doc", "welcome", "talk2me", "fatuorum", "contribs"])}

    def glove_preprocess(self, text):
    # Different regex parts for smiley faces
        eyes = "[8:=;]"
        nose = "['`\-]?"
        text = re.sub("https?:* ", "<URL>", text)
        text = re.sub("www.* ", "<URL>", text)
        text = re.sub("\[\[User(.*)\|", '<USER>', text)
        text = re.sub("<3", '<HEART>', text)
        text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
        text = re.sub(eyes + nose + "[Dd)]", '<SMILE>', text)
        text = re.sub("[(d]" + nose + eyes, '<SMILE>', text)
        text = re.sub(eyes + nose + "p", '<LOLFACE>', text)
        text = re.sub(eyes + nose + "\(", '<SADFACE>', text)
        text = re.sub("\)" + nose + eyes, '<SADFACE>', text)
        text = re.sub(eyes + nose + "[/|l*]", '<NEUTRALFACE>', text)
        text = re.sub("/", " / ", text)
        text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
        text = re.sub("([!]){2,}", "! <REPEAT>", text)
        text = re.sub("([?]){2,}", "? <REPEAT>", text)
        text = re.sub("([.]){2,}", ". <REPEAT>", text)
        pattern = re.compile(r"(.)\1{2,}")
        text = pattern.sub(r"\1" + " <ELONG>", text)

        return text

    def clean_text(self, s):
        s = s.lower()
        # 2. Drop \n and  \t
        s = s.replace(r"\n", r" ")
        s = s.replace(r"\t", r" ")
        s = s.replace(r"\b", r" ")
        s = s.replace(r"\r", r" ")

        # Drop numbers - as a scientist I don't think numbers are toxic ;-)
        s = re.sub(r"\d+", r" ", s)
        # Capture IP address
        s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
        # Isolate punctuation
        s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
        # Remove some special characters
        s = re.sub(r'([\;\:\|•«\n])', ' ', s)
        # clean bad words
        s = re.sub(" u ", "you", s)
        s = re.sub("\nu ", "you", s)
        s = re.sub(" u\n", "you", s)
        s = re.sub("fucksex", "fuck sex", s)

        template = re.compile(r"([a-z])(\1{3,})")
        s = re.sub(template, r" \1 ", s)
        # Remove extra spaces - At the end of previous operations we
        # multiplied space accurences
        s = re.sub(r'[ ]+', r" ", s)
        # Remove ending space if any
        s = re.sub(r'[ ]+$', "", s)
        return s

    def remove_stopwords(self, lt):
        new_data = []
        for i in lt:
            arr = str(i).split()
            xx = ""
            for j in arr:
                j = str(j).lower()
                if j[:4] is "http" or j[:3] is "www":
                    continue
                if j in self.keys:
                    j = self.repl[j]
                if j in self.stopwords:
                    j = ""
                xx += j + " "
            new_data.append(xx)
        return new_data

    def count_regexp_occ(self, regexp="", text=None):
        """ Simple way to get the number of occurence of a regex"""
        return len(re.findall(regexp, text))

    def get_num_features(self, df):
        """
        Check all sorts of content as it may help find toxic comment
        Though I'm not sure all of them improve scores
        """
        # Count number of \n
        df["ant_slash_n"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"\n", x))
        # Get length in words and characters
        df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
        df["total_length"] = df["comment_text"].apply(lambda x: len(x))
        df['capitals'] = df['comment_text'].apply(
            lambda comment: sum(1 for c in comment if c.isupper()))
        df['caps_vs_length'] = df.apply(lambda row: float(
            row['capitals']) / float(row['total_length']), axis=1)

        # Number of unique words
        df['num_unique_words'] = df['comment_text'].apply(
            lambda comment: len(set(w for w in comment.split())))
        # Check number of upper case, if you're angry you may write in upper
        # case
        df["nb_upper"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"[A-Z]", x))
        # Number of F words - f..k contains folk, fork,
        df["nb_fk"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
        # Number of S word
        df["nb_sk"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
        # Number of D words
        df["nb_dk"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"[dD]ick", x))
        # Number of occurence of You, insulting someone usually needs someone
        # called : you
        df["nb_you"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"\W[Yy]ou\W", x))
        # Numer of ! and ?
        df["nb_exclamation_marks"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"[?!]", x))
        df['num_punctuation'] = df['comment_text'].apply(
            lambda comment: sum(comment.count(w) for w in '.,;:'))
        # Just to check you really refered to my mother ;-)
        df["nb_mother"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"\Wmother\W", x))
        # Just checking for toxic 19th century vocabulary
        df["nb_ng"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"\Wnigger\W", x))
        # Some Sentences start with a <:> so it may help
        df["start_with_columns"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"^\:+", x))
        # Check for time stamp
        df["has_timestamp"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"\d{2}|:\d{2}", x))
        # Check for dates 18:44, 8 December 2010
        df["has_date_long"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
        # Check for date short 8 December 2010
        df["has_date_short"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
        # Check for http links
        df["has_http"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"http[s]{0,1}://\S+", x))
        # check for mail
        df["has_mail"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(
                r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
        )
        # Looking for words surrounded by == word == or """" word """"
        df["has_emphasize_equal"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"\={2}.+\={2}", x))
        df["has_emphasize_quotes"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r"\"{4}\S+\"{4}", x))

    def transform_num_features(self, data, is_nn_model):
        # Scaling numerical features with MinMaxScaler though tree boosters
        # don't need that
        class_names = ['toxic', 'severe_toxic', 'obscene',
                       'threat', 'insult', 'identity_hate']

        num_features = [f_ for f_ in data.columns
                        if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                      'has_ip_address', "raw_word_len"] + class_names]

        skl = MinMaxScaler()
        if is_nn_model is not True:
            data_num_features = csr_matrix(
                skl.fit_transform(data[num_features]))
            data_num_features.columns = num_features
        else:
            data_num_features = skl.fit_transform(data[num_features])

        return data_num_features
        
class BaseTokenizer(object):
    def process_text(self, text):
        raise NotImplemented

    def process(self, texts):
        for text in texts:
            yield self.process_text(text)


RE_PATTERNS = {
    ' american ':
        [
            'amerikan'
        ],

    ' adolf ':
        [
            'adolf'
        ],


    ' hitler ':
        [
            'hitler'
        ],

    ' fuck':
        [
            '(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
            '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',
            ' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k', 'f u u c',
            '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*',
            'feck ', ' fux ', 'f\*\*', 
            'f\-ing', 'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck'
        ],

    ' ass ':
        [
            '[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\$\$'
                                                           '[^a-z]anus', ' a\*s\*s', '[^a-z]ass[^a-z ]',
            'a[@#\$%\^&\*][@#\$%\^&\*]', '[^a-z]anal ', 'a s s'
        ],

    ' ass hole ':
        [
            ' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\$\$hole'
        ],

    ' bitch ':
        [
            'b[w]*i[t]*ch', 'b!tch',
            'bi\+ch', 'b!\+ch', '(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)',
            'biatch', 'bi\*\*h', 'bytch', 'b i t c h'
        ],

    ' bastard ':
        [
            'ba[s|z]+t[e|a]+rd'
        ],

    ' trans gender':
        [
            'transgender'
        ],

    ' gay ':
        [
            'gay'
        ],

    ' cock ':
        [
            '[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',
            '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'
        ],

    ' dick ':
        [
            ' dick[^aeiou]', 'deek', 'd i c k'
        ],

    ' suck ':
        [
            'sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'
        ],

    ' cunt ':
        [
            'cunt', 'c u n t'
        ],

    ' bull shit ':
        [
            'bullsh\*t', 'bull\$hit'
        ],

    ' homo sex ual':
        [
            'homosexual'
        ],

    ' jerk ':
        [
            'jerk'
        ],

    ' idiot ':
        [
            'i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots'
                                                                                      'i d i o t'
        ],

    ' dumb ':
        [
            '(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'
        ],

    ' shit ':
        [
            'shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\$hit', 's h i t'
        ],

    ' shit hole ':
        [
            'shythole'
        ],

    ' retard ':
        [
            'returd', 'retad', 'retard', 'wiktard', 'wikitud'
        ],

    ' rape ':
        [
            ' raped'
        ],

    ' dumb ass':
        [
            'dumbass', 'dubass'
        ],

    ' ass head':
        [
            'butthead'
        ],

    ' sex ':
        [
            'sexy', 's3x', 'sexuality'
        ],


    ' nigger ':
        [
            'nigger', 'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'
        ],

    ' shut the fuck up':
        [
            'stfu'
        ],

    ' pussy ':
        [
            'pussy[^c]', 'pusy', 'pussi[^l]', 'pusses'
        ],

    ' faggot ':
        [
            'faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',
            '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot',
        ],

    ' mother fucker':
        [
            ' motha ', ' motha f', ' mother f', 'motherucker',
        ],

    ' whore ':
        [
            'wh\*\*\*', 'w h o r e'
        ],
}


class PatternTokenizer(BaseTokenizer):
    def __init__(self, lower=True, initial_filters=r"[^a-z0-9!@#\$%\^\&\*_\-,\.' ]", patterns=RE_PATTERNS,
                 remove_repetitions=True):
        self.lower = lower
        self.patterns = patterns
        self.initial_filters = initial_filters
        self.remove_repetitions = remove_repetitions

    def process_text(self, text):
        x = self._preprocess(text)
        for target, patterns in self.patterns.items():
            for pat in patterns:
                x = re.sub(pat, target, x)
        x = re.sub(r"[^a-z' ]", ' ', x)
        return x.split()

    def process_ds(self, ds):
        ### ds = Data series

        # lower
        ds = copy.deepcopy(ds)
        if self.lower:
            ds = ds.str.lower()
        # remove special chars
        if self.initial_filters is not None:
            ds = ds.str.replace(self.initial_filters, ' ')
        # fuuuuck => fuck
        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 
            ds = ds.str.replace(pattern, r"\1")

        for target, patterns in self.patterns.items():
            for pat in patterns:
                ds = ds.str.replace(pat, target)

        ds = ds.str.replace(r"[^a-z' ]", ' ')

        return ds.str.split()

    def _preprocess(self, text):
        # lower
        if self.lower:
            text = text.lower()

        # remove special chars
        if self.initial_filters is not None:
            text = re.sub(self.initial_filters, ' ', text)

        # fuuuuck => fuck
        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
            text = pattern.sub(r"\1", text)
        return text        
