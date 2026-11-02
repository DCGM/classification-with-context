"""Additive keyword groups over the existing raw stats schema; no new extraction.

These semantic subsets intentionally overlap the original coarse groups. Each
adds whole-page log counts plus header/footer presence, in the listed order.
"""

FINE_KEYWORD_GROUPS = {
    'volume': ('dil ', 'band ', 'volume', 'tome', 'tomus', 'pars ', 'kotet', 'том ', 'tomo', 'bind ', 'svazek'),
    'instalment': ('sesit', 'heft', 'lieferung', 'livraison', 'fascicule', 'fasciculus', 'zeszyt',
                  'fuzet', 'выпуск', 'fascicolo', 'cuaderno', 'hefte'),
    'place': ('praha', 'v praze', 'brno', 'wien', 'leipzig', 'prag', 'london', 'new york', 'paris',
              'pragae', 'viennae', 'warszawa', 'krakow', 'budapest', 'москва', 'петербург',
              'roma', 'milano', 'madrid', 'københavn'),
    'byline': ('napsal', 'sepsal', 'usporadal', 'upravil', 'von ', 'herausgegeben', 'verfasst',
               'bearbeitet', 'edited by', 'par ', 'auctore', 'napisał', 'irta', 'под редакциеи'),
    'translation': ('prelozil', 'ubersetzt', 'translated', 'traduit', 'przełozył', 'forditotta',
                    'перевод', 'tradotto', 'tradotta', 'traducido', 'traduccion', 'oversat'),
    'printer': ('vytiskl', 'tiskem', 'tiskarn', 'knihtiskarn', 'vytisk', 'sazb', 'gedruckt',
                'druck von', 'druck und verlag', 'buchdruckerei', 'printed in', 'printed by',
                'imprime', 'imprimerie', 'acheve d imprimer', 'typis ', 'excudit', 'typographia',
                'drukarnia', 'nyomda', 'nyomatott', 'типограф', 'отпечатано', 'stampato',
                'tipografia', 'impreso', 'imprenta', 'trykt', 'bogtrykkeri'),
    'rights': ('isbn', 'issn', 'copyright', 'prava vyhrazena', 'vsechna prava',
               'alle rechte vorbehalten', 'all rights reserved', 'tous droits reserves',
               'prawa zastrzezone', 'minden jog fenntartva', 'tutti i diritti',
               'derechos reservados', 'eftertryk'),
    'toc_heading': ('obsah', 'prehled', 'inhalt', 'contents', 'table of contents', 'table des matieres',
                    'sommaire', 'table analytique', 'index capitum', 'elenchus', 'conspectus',
                    'spis tresci', 'spis rzeczy', 'tartalom', 'содержание', 'оглавление',
                    'indice', 'sommario', 'tabla de materias', 'indhold'),
    'toc_column': ('strana', 'stranka', 'str ', 'seite', 'page ', 'pages', 'pag ', 'oldal', 'стр ', 'side '),
    'index_kind': ('jmenny', 'vecny', 'osobnich', 'mistnich', 'jmen osob', 'seznam jmen',
                   'sachregister', 'namenregister', 'personenregister', 'ortsregister',
                   'sachverzeichnis', 'namensverzeichnis', 'ortsnamen', 'names of persons',
                   'subject index', 'noms de personnes', 'noms de lieux', 'index rerum',
                   'index nominum', 'nominum', 'locorum', 'osobowy', 'nazw miejscowych',
                   'nevmutato', 'targymutato', 'helynev', 'именнои', 'предметныи',
                   'indice analitico', 'indice dei nomi', 'nomi di persona', 'onomastic',
                   'navneregister', 'stikordsregister', 'personregister'),
    'end_word': ('konec ', 'schluss ', 'the end', 'fin ', 'finis', 'explicit', 'laus deo',
                 'koniec', 'vege', 'конец', 'fine ', 'slut'),
}
