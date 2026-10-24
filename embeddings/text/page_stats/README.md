# Page statistics

A non-neural page embedding, used next to the text (mmBERT, Qwen3) and image (DINOv2) embeddings. It is
built from a page's OCR, ALTO by default and plain text as a fallback. It contains:
- keyword hits;
- regex patterns;
- line statistics;
- the ALTO line geometry.

Everything lives in `page_stats.py`, which needs only numpy and the standard library.

- **Extraction:** OCR → *full vector*. This is what gets stored, sparse (`to_sparse()`), e.g. in an
  LMDB keyed like the text embeddings.
- **Views:** stored full vectors → what a model gets. Keywords per stem or grouped, and a selection of
  blocks or features.

```python
from embeddings.text.page_stats import PageStats

stats = PageStats()
full = stats.embed_alto(alto_bytes)                  # one page: len(FEATURES) numbers (float32)
full = stats.embed(alto_list, workers=8)             # many pages: one row per page
full = stats.embed(text_list, kind='text')           # fallback without ALTO: the ALTO features are 0

x, names = stats.view(full)                          # all features, keyword counts as log(1 + n)
x, names = stats.view(full, grouped=True)            # keyword groups (GROUPED_FEATURES)
x, names = stats.view(full, grouped=True, exclude=['alto_word_conf', 'alto_word_boxes'])
x, names = stats.view(full, include=['keywords', 'alto_lines'], exclude=['kw:viz '])
x, names = PageStats(groups=LANGUAGE_GROUPS).view(full, grouped=True, include=['kw_*'])  # per language
stats.metadata()                                     # version, feature names, keyword groups: store with the vectors

value = to_sparse(full_row)                          # what goes into the LMDB
full = stats.decode([value1, value2])                # stored values -> full vectors, then view()
```

`include` and `exclude` take block names, feature names or fnmatch patterns (`'head_*'`,
`'kw:obsah'`). A stored vector is checked against the feature count of the current version.

From the command line, `python -m embeddings.text.page_stats.page_stats page.xml [--grouped]` prints a
page's nonzero features. `--keyword-table` prints the keyword tables below.

## The full vector

| Block | Features | Values |
|---|---|---|
| `keywords` | `kw:<stem>` | hits of the stem on the whole page (count) |
| `head_keywords` | `head_kw:<stem>` | hits in the header: the first 3 non-empty lines (count) |
| `foot_keywords` | `foot_kw:<stem>` | hits in the footer: the last 3 non-empty lines, none on pages of up to 3 lines (count) |
| `patterns` | `pat:<pattern>` | lines matching a regex (count): `isbn`, `price`, `year`, `roman_year`, `scale`, `page_ref`, `leader_number`, `figure_ref` |
| `head_patterns` | `head_pat:<pattern>` | header lines matching it (count) |
| `foot_patterns` | `foot_pat:<pattern>` | footer lines matching it (count) |
| `text_lines` | 22 | line statistics of the text (below) |
| `alto_lines` | 27 | ALTO features from the line boxes (below) |
| `alto_word_conf` | 2 | `conf_mean`, `conf_low`: the mean word confidence, and the share of words under 0.5 |
| `alto_word_boxes` | 3 | `word_gap_med`, `wide_gap_lines`, `char_width_ratio`: from the word boxes |

That is 3 × the number of stems + 24 + 22 + 32 numbers.

**Counts are stored raw**, so the stems can be regrouped and summed later. `view()` log-scales them
(`log_counts=True`), and the grouped view turns the header/footer counts into 0/1 flags, as in the
experiments. Everything else is stored as computed: ratios, shares of lines, log counts and small
averages.

**Storage is sparse.** A page has a few dozen nonzero numbers out of over 3,000. `to_sparse()` stores
the uint16 indices of the nonzero entries followed by their float16 values (little endian, 4 bytes per
nonzero), and `from_sparse()` / `PageStats.decode()` restore the full vector. On 5,000 public pages
float16 kept every value within 0.05% (at most 0.6% of a feature's spread). Counts are exact up to
2,048, and values are clipped at 65,504.

**Text lines.** With ALTO, the text is the ALTO line texts sorted top to bottom; with plain text, its
lines. The 21 `text_lines` features:
- `log_chars`, `log_lines`, `empty`;
- line lengths: `mean_line_len`, `std_line_len`, `short_lines` (< 20 characters);
- shares of lines:
  - ending in a number (`lines_end_number`);
  - that are a number only (`number_only_lines`);
  - index-like `word, 12` (`index_lines`);
  - with a dot leader (`dot_leader_lines`);
  - starting a list item (`list_start_lines`);
  - in capitals (`allcaps_lines`);
  - mostly digits (`numeric_lines`);
  - with 3+ numbers (`multi_number_lines`);
- character shares: `digit_ratio`, `upper_ratio`, `punct_ratio`, `alpha_ratio`;
- `words_per_line`;
- `alphabetical_lines`: line starts in alphabetical order, as in an index;
- `first_line_caps`;
- `short_words`: the share of words of up to 2 characters (OCR noise).

**ALTO lines.** The 27 `alto_lines` features:
- `page_aspect`, and log counts of `n_blocks`, `n_lines`, `n_words`;
- the text box: `text_area`, `text_top`, `text_bottom`, `text_left`, `text_right`, `text_center_x`,
  `text_center_y`, as fractions of the page;
- line heights:
  - `line_height_med` (relative to the page);
  - `line_height_max_ratio`;
  - `large_lines` (> 1.5 × the median);
  - `top_line_ratio`, `top_line_centered` (a large centred heading);
- line layout:
  - shares of `centered_lines`, `right_aligned_lines`, `indented_lines`, `full_width_lines`;
  - `columns`: clusters of line left edges;
  - `line_gap_med`, `line_gap_max` (in median line heights);
  - `right_numbers`: lines ending in a right-aligned number, as in a TOC;
- shares of lines in the top and bottom 15% of the page: `top_band_lines`, `bottom_band_lines`;
- `has_ocr_xml`: 1 for ALTO, 0 for the text fallback.

The word confidences and word boxes exist only in ALTO from OCR that produces them. In the private
storage, the ALTO was converted from PAGE XML (`embedding_probe/convert_page_xml_to_alto.py`): its
words are placed along the line by their length, and every word gets its line's confidence. On the
public development sample neither word block improved the classifier (weighted F1 0.613–0.628 vs
0.620 for the lines alone). They are stored so they can be tried again, not used by default.

## Keywords

**Matching.** A character trie of the stems is walked from every word start, in one pass over the text
(`stem_counts()`). Text and stems are compared lower case, without diacritics (`fold()`). Inside a line,
punctuation counts as a space. A stem matches at the start of a word, so it covers the inflected
forms: `predmluv` matches předmluva, předmluvy and předmluvou.
- A stem with a space is a phrase.
- A trailing space ends the word: `viz ` does not match `vizitka`.
- Stems are written in the folded form, which `_check_stems()` enforces at import:
  - Polish `ł` and Danish `æ`, `ø` stay as they are;
  - Cyrillic `й` becomes `и`;
  - `ß` becomes `ss`.

**Languages.** The stems cover the union of the languages they were written for: Czech, German,
English, French, Latin, Polish, Hungarian, Russian, Italian, Spanish and Danish (`LANGUAGES`, ISO 639-2
as in the library metadata). Each group has stems in each of them, except where the page type does not
occur in a language, e.g. advertisements in Latin. `any` holds language-neutral stems: ISBN, brand
names on calibration targets, Italian music terms.

Left out are stems that are frequent words of another language, especially Czech, which most pages
are in: `by `, `tom `, `ano `, `a mi `, `a ma `.

**Groups.** Each stem belongs to one or more groups (`KEYWORD_GROUPS`), and a group targets page
types. A stem in several groups or languages is extracted once. The grouping is applied only in
`view()`, so it can change without re-extraction:
- `PageStats(groups={...})` takes any grouping of existing stems;
- `LANGUAGE_GROUPS` groups the stems by language, as a rough language profile.

A *new* stem, however, needs re-extraction.

**Where the stems come from:**
- **The original 23 groups:** designed on public text, on the annotated training pages and the
  DB-labelled public development sample (`embedding_probe/explore_public_text.py`,
  `cue_dev_report.py`). In the experiments they scored weighted F1 0.651 → 0.713 (text cues alone
  → + ALTO lines, K = 2) and 0.811 next to mmBERT + DINOv2.
- **Extended in version 1:**
  - every group was made language-symmetric;
  - stems were added from a varied sample of public pages of documents not used before
    (`build_keyword_sample.py`, `keyword_candidates.py`, `front_back_headings.py`), for example:
    - TOC page-column and chapter words;
    - kinds of index;
    - the reader addressed in prefaces;
    - table column words;
    - periodical subscription terms;
    - annual reports and programmes on covers;
    - price lists and catalogues in advertisements;
    - Latin dedication formulas.

  Public text is mostly Czech and German, so the stems of the other languages are from language
  knowledge rather than from data.
- **How often each stem fires** on public pages: `stem_frequencies.md`
  (`embedding_probe/stem_frequency_report.py`).

**Private pages.** The full vector of a private page is derived from its text. The per-stem counts in
particular are a small bag of words. See the repository's `CLAUDE.md` before storing or inspecting the
vectors of private pages.

<!-- keyword-table: generated by `python -m embeddings.text.page_stats.page_stats --keyword-table`; a test checks it -->

#### toc

| language | stems |
|---|---|
| ces | `obsah`, `kapitola`, `kap `, `strana`, `stranka`, `str `, `prehled` |
| deu | `inhalt`, `kapitel`, `abschnitt`, `seite` |
| eng | `contents`, `table of contents`, `chapter`, `page ` |
| fra | `table des matieres`, `sommaire`, `chapitre`, `table analytique`, `pages` |
| lat | `index capitum`, `elenchus`, `conspectus`, `caput`, `cap `, `pag ` |
| pol | `spis tresci`, `spis rzeczy`, `rozdział`, `str ` |
| hun | `tartalom`, `fejezet`, `oldal` |
| rus | `содержание`, `оглавление`, `глава`, `стр ` |
| ita | `indice`, `sommario`, `capitolo`, `pag ` |
| spa | `indice`, `tabla de materias`, `capitulo`, `pag ` |
| dan | `indhold`, `kapitel`, `side ` |

#### index

| language | stems |
|---|---|
| ces | `rejstrik`, `registrik`, `rejstr`, `jmenny`, `vecny`, `ukazatel`, `viz `, `abecedn`, `osobnich`, `mistnich`, `jmen osob`, `seznam jmen` |
| deu | `register`, `sachregister`, `namenregister`, `personenregister`, `ortsregister`, `sachverzeichnis`, `namensverzeichnis`, `siehe `, `alphabetisch`, `personen`, `ortsnamen` |
| eng | `index`, `see also`, `alphabetical`, `names of persons`, `subject index` |
| fra | `index`, `table alphabetique`, `voir `, `alphabetique`, `noms de personnes`, `noms de lieux` |
| lat | `index`, `index rerum`, `index nominum`, `vide `, `alphabetic`, `nominum`, `locorum` |
| pol | `indeks`, `skorowidz`, `zob `, `alfabetyczn`, `osobowy`, `nazw miejscowych` |
| hun | `mutato`, `nevmutato`, `targymutato`, `lasd `, `beturend`, `helynev` |
| rus | `указатель`, `см `, `алфавит`, `именнои`, `предметныи` |
| ita | `indice analitico`, `indice dei nomi`, `vedi `, `alfabetic`, `nomi di persona` |
| spa | `indice alfabetico`, `vease`, `alfabetic`, `onomastic` |
| dan | `register`, `navneregister`, `stikordsregister`, `alfabetisk`, `personregister` |

#### errata

| language | stems |
|---|---|
| ces | `oprav`, `tiskov`, `omyl`, `radek`, `shora`, `zdola`, `misto `, `cti ` |
| deu | `druckfehler`, `berichtigung`, `verbesserung`, `lies `, `statt `, `zeile` |
| eng | `errata`, `corrections`, `misprint`, `additions` |
| fra | `errata`, `fautes`, `lisez`, `au lieu de`, `ligne` |
| lat | `errata`, `corrigenda`, `emendanda`, `lege ` |
| pol | `errata`, `błedy`, `sprostowan`, `zamiast`, `czytaj`, `wiersz` |
| hun | `sajtohib`, `helyesbit`, `olvasd`, `helyett` |
| rus | `опечатк`, `исправлен`, `напечатано`, `следует читать`, `строк` |
| ita | `errata corrige`, `correzion`, `leggi `, `invece di`, `riga` |
| spa | `fe de erratas`, `erratas`, `debe decir`, `linea` |
| dan | `rettelser`, `trykfejl`, `læs `, `linje` |

#### preface

| language | stems |
|---|---|
| ces | `predmluv`, `predml`, `uvodem`, `uvod`, `ctenar`, `laskav` |
| deu | `vorrede`, `vorwort`, `einleitung`, `geleitwort`, `leser` |
| eng | `preface`, `introduction`, `foreword`, `reader` |
| fra | `preface`, `avant propos`, `introduction`, `avertissement`, `lecteur` |
| lat | `praefatio`, `prefatio`, `prooemium`, `ad lectorem`, `lectori`, `lector`, `benevole` |
| pol | `przedmow`, `wstep`, `słowo wstepne`, `czytelnik` |
| hun | `eloszo`, `bevezet`, `olvaso` |
| rus | `предислов`, `введение`, `вступлен`, `читател` |
| ita | `prefazione`, `introduzione`, `premessa`, `lettore` |
| spa | `prologo`, `prefacio`, `introduccion`, `lector` |
| dan | `forord`, `indledning`, `læser` |

#### dedication

| language | stems |
|---|---|
| ces | `venovan`, `venuj`, `pamatce`, `na pamet`, `drahe`, `milovan` |
| deu | `gewidmet`, `zugeeignet`, `meinem`, `meiner`, `zum andenken`, `gedachtnis` |
| eng | `dedicated`, `in memory`, `in loving memory` |
| fra | `dedie`, `a la memoire` |
| lat | `dedicatoria`, `dedicat`, `illustrissim`, `reverendissim`, `excellentissim`, `memoriae`, `honori`, `venerationi`, `cliens`, `devotissim`, `humillim`, `obsequentissim`, `majestat` |
| pol | `poswiec`, `pamieci`, `mojej`, `mojemu` |
| hun | `ajanl`, `emlekere`, `emlekenek`, `szuleimnek` |
| rus | `посвящ`, `памяти` |
| ita | `dedicat`, `alla memoria` |
| spa | `dedicad`, `a la memoria` |
| dan | `tilegnet`, `til minde` |

#### imprimatur

| language | stems |
|---|---|
| ces | `cirkevni schval`, `schvaleno`, `ordinariat` |
| deu | `druckerlaubnis`, `kirchliche druckerlaubnis`, `genehmig`, `zensur`, `ordinariat` |
| eng | `ecclesiastical approval`, `permission`, `censor` |
| fra | `permis`, `approbation`, `censure` |
| lat | `imprimatur`, `nihil obstat`, `approbat`, `privileg`, `censor`, `licentia`, `facultas`, `admittitur`, `typis `, `vicarius`, `ordinariat` |
| pol | `pozwolen`, `zatwierdz`, `cenzur` |
| hun | `jovahagy`, `egyhazi jovahagy`, `engedely` |
| rus | `дозволено`, `цензур` |
| ita | `si stampi`, `si permette`, `approvazione`, `revisore` |
| spa | `puede imprimirse`, `licencia`, `aprobacion`, `censura` |
| dan | `godkendt`, `tilladelse` |

#### imprint

| language | stems |
|---|---|
| any | `isbn`, `issn`, `copyright` |
| ces | `vytiskl`, `tiskem`, `tiskarn`, `knihtiskarn`, `vytisk`, `vydal`, `nakladem`, `naklad `, `nakladatel`, `vydani`, `odpovedny redaktor`, `redaktor`, `sazb`, `autorskych arch`, `autorske arch`, `prava vyhrazena`, `vsechna prava`, `brozovan`, `vazan`, `tematicka skupina`, `povoleno`, `schvalil`, `edice`, `svazek` |
| deu | `gedruckt`, `druck von`, `druck und verlag`, `verlag`, `alle rechte vorbehalten`, `auflage`, `buchdruckerei` |
| eng | `printed in`, `printed by`, `all rights reserved`, `published by`, `publisher`, `edition` |
| fra | `imprime`, `imprimerie`, `tous droits reserves`, `editeur`, `acheve d imprimer` |
| lat | `typis `, `impensis`, `apud `, `excudit`, `typographia` |
| pol | `drukarnia`, `wydawnictw`, `nakład`, `wydanie`, `prawa zastrzezone` |
| hun | `nyomda`, `nyomatott`, `kiado`, `kiadas`, `minden jog fenntartva` |
| rus | `типограф`, `издательств`, `тираж`, `подписано`, `отпечатано` |
| ita | `stampato`, `tipografia`, `editore`, `edizione`, `tutti i diritti` |
| spa | `impreso`, `imprenta`, `editorial`, `edicion`, `derechos reservados` |
| dan | `trykt`, `bogtrykkeri`, `forlag`, `udgave`, `eftertryk` |

#### bibliography

| language | stems |
|---|---|
| ces | `bibliografi`, `literatura`, `prameny`, `pramen`, `seznam literatury`, `pouzita literatura`, `citovan` |
| deu | `literatur`, `literaturverzeichnis`, `quellen`, `bibliographie`, `schrifttum` |
| eng | `bibliography`, `references`, `literature`, `sources`, `works cited` |
| fra | `bibliographie`, `references`, `sources`, `ouvrages cites` |
| lat | `bibliographia`, `fontes` |
| pol | `bibliografi`, `literatura`, `zrodła`, `pismiennictw` |
| hun | `irodalom`, `forrasok`, `bibliografi` |
| rus | `библиограф`, `литература`, `источник` |
| ita | `bibliografi`, `fonti`, `riferimenti` |
| spa | `bibliografi`, `referencias`, `fuentes` |
| dan | `litteratur`, `kilder`, `bibliografi` |

#### list_illustrations

| language | stems |
|---|---|
| ces | `seznam vyobrazeni`, `seznam ilustraci`, `seznam obrazk`, `seznam obrazu`, `vyobrazeni`, `ilustrac`, `obrazk`, `obr `, `podobizn`, `obrazy v textu`, `prilohy` |
| deu | `abbildung`, `tafel`, `bilder`, `illustration`, `bildnis`, `tafeln`, `abbildungen im text` |
| eng | `illustrations`, `list of illustrations`, `figures`, `fig `, `plates`, `portrait`, `frontispiece` |
| fra | `planche`, `figur`, `table des figures`, `gravure`, `portrait`, `hors texte` |
| lat | `figur`, `tabula`, `effigies`, `tabulae` |
| pol | `ilustracj`, `spis ilustracji`, `rycin`, `portret`, `tablic` |
| hun | `abrak`, `kepek`, `illusztrac`, `arckep`, `kepmelleklet` |
| rus | `иллюстрац`, `рисунк`, `рис `, `портрет`, `таблиц` |
| ita | `illustrazion`, `tavol`, `figure`, `ritratto`, `fuori testo` |
| spa | `ilustracion`, `laminas`, `grabado`, `retrato`, `fuera de texto` |
| dan | `illustrationer`, `billeder`, `afbildning`, `portræt`, `tavler` |

#### list_maps

| language | stems |
|---|---|
| ces | `seznam map`, `mapy`, `plany`, `planu` |
| deu | `karten`, `plane`, `kartenverzeichnis` |
| eng | `maps`, `map `, `plans` |
| fra | `cartes`, `plans` |
| lat | `tabulae geographicae`, `mappa` |
| pol | `mapy`, `spis map`, `plany` |
| hun | `terkep` |
| rus | `карты`, `карта `, `планы` |
| ita | `carte`, `mappe`, `piante` |
| spa | `mapas`, `planos` |
| dan | `kort `, `kortene` |

#### list_tables

| language | stems |
|---|---|
| ces | `seznam tabulek`, `tabulek`, `tabulky`, `prehled tabul` |
| deu | `tabellen`, `tabellenverzeichnis` |
| eng | `tables`, `list of tables` |
| fra | `tableaux` |
| lat | `tabulae` |
| pol | `tabel`, `spis tabel` |
| hun | `tablazat` |
| rus | `таблиц` |
| ita | `tabelle`, `tavole` |
| spa | `tablas`, `cuadros` |
| dan | `tabeller` |

#### appendix

| language | stems |
|---|---|
| ces | `priloh`, `dodatek`, `dodatk` |
| deu | `anhang`, `beilage`, `nachtrag` |
| eng | `appendix`, `supplement`, `addendum` |
| fra | `annexe`, `appendice`, `supplement` |
| lat | `appendix`, `additamentum`, `supplement` |
| pol | `aneks`, `dodatek`, `załacznik` |
| hun | `fuggelek`, `melleklet` |
| rus | `приложени`, `дополнени` |
| ita | `appendice`, `supplemento`, `allegat` |
| spa | `apendice`, `anexo`, `suplemento` |
| dan | `tillæg`, `bilag` |

#### abstract

| language | stems |
|---|---|
| ces | `souhrn`, `abstrakt`, `klicova slova`, `klicova`, `resume` |
| deu | `zusammenfassung`, `schlusselworter` |
| eng | `abstract`, `summary`, `keywords` |
| fra | `resume`, `mots cles` |
| lat | `summarium`, `argumentum` |
| pol | `streszczen`, `słowa kluczowe` |
| hun | `osszefoglal`, `kivonat`, `kulcsszavak` |
| rus | `резюме`, `аннотац`, `ключевые слова` |
| ita | `riassunto`, `parole chiave` |
| spa | `resumen`, `palabras clave` |
| dan | `resume`, `sammendrag`, `nøgleord` |

#### advertisement

| language | stems |
|---|---|
| ces | `cena `, `k dostani`, `prodej`, `prodava`, `nabizi`, `objednav`, `adresa`, `telefon`, `tel `, `inzer`, `firma`, `sklad`, `zdarma`, `levne`, `doporucuje`, `zavod`, `franko`, `zbozi`, `cenik`, `tovarn`, `na pozadani`, `nejlevn`, `prvotridn`, `objedna`, `vyslo`, `knihkupectv` |
| deu | `inserat`, `anzeige`, `verkauf`, `zu haben`, `preis`, `vermiet`, `zu verkaufen`, `empfiehlt`, `bestell`, `erhaltlich`, `billig`, `preisliste`, `katalog`, `fabrik`, `franko`, `erbeten`, `auf verlangen` |
| eng | `advertis`, `for sale`, `price`, `cheap`, `apply to`, `sold by`, `catalogue`, `price list`, `manufactur`, `post free`, `on application` |
| fra | `prix`, `vente`, `a vendre`, `annonce`, `commande`, `adresse`, `catalogue`, `prix courant`, `fabrique`, `franco`, `sur demande` |
| pol | `cena `, `sprzedaz`, `ogłoszen`, `polec`, `zamowien`, `cennik`, `katalog`, `fabryk`, `franco`, `na zadanie` |
| hun | `kaphato`, `elado`, `hirdet`, `arjegyzek`, `katalogus`, `gyar `, `bermentve` |
| rus | `цена`, `продаж`, `объявлен`, `продается`, `преискурант`, `каталог`, `фабрик`, `по требованию` |
| ita | `prezzo`, `vendita`, `vendesi`, `inserzion`, `listino`, `catalogo`, `fabbrica`, `franco`, `a richiesta` |
| spa | `precio`, `venta`, `se vende`, `anuncio`, `catalogo`, `lista de precios`, `fabrica`, `franco`, `a peticion` |
| dan | `pris `, `salg`, `annonce`, `sælges`, `prisliste`, `katalog`, `fabrik`, `franko` |

#### title

| language | stems |
|---|---|
| ces | `napsal`, `sepsal`, `prelozil`, `usporadal`, `upravil`, `vydava`, `dil `, `praha`, `v praze`, `brno` |
| deu | `von `, `herausgegeben`, `ubersetzt`, `verfasst`, `bearbeitet`, `band `, `wien`, `leipzig`, `prag` |
| eng | `edited by`, `translated`, `volume`, `london`, `new york` |
| fra | `par `, `traduit`, `tome`, `paris` |
| lat | `auctore`, `editio`, `tomus`, `pars `, `pragae`, `viennae` |
| pol | `napisał`, `przełozył`, `warszawa`, `krakow` |
| hun | `irta`, `forditotta`, `kotet`, `budapest` |
| rus | `перевод`, `под редакциеи`, `том `, `москва`, `петербург` |
| ita | `tradotto`, `tradotta`, `volume`, `roma`, `milano` |
| spa | `traducido`, `traduccion`, `tomo`, `madrid` |
| dan | `oversat`, `udgivet`, `bind `, `københavn` |

#### masthead

| language | stems |
|---|---|
| ces | `rocnik`, `cislo`, `predplatn`, `vychazi`, `administrac`, `redakce`, `vychaz`, `ctvrtletn`, `pulletn`, `celoletn`, `predplac`, `jednotliv`, `nefrankovan`, `rukopisy se nevraceji`, `inserty`, `reklamace`, `tydennik`, `dennik`, `casopis` |
| deu | `jahrgang`, `nummer`, `zeitung`, `blatt`, `abonnement`, `bezugspreis`, `erscheint`, `monatlich`, `vierteljahr`, `halbjahr`, `ganzjahr`, `taglich`, `zustellung`, `pranumeration`, `einzelpreis`, `einzelnummer`, `wochenschrift`, `zeitschrift` |
| eng | `subscription`, `published weekly`, `published monthly`, `issue`, `per annum`, `weekly`, `monthly`, `quarterly`, `single copy`, `journal` |
| fra | `abonnement`, `annee`, `numero`, `parait`, `par an`, `hebdomadaire`, `mensuel`, `trimestre`, `revue`, `journal` |
| lat | `annus`, `periodic` |
| pol | `rocznik`, `numer`, `prenumerat`, `wychodzi`, `redakc`, `kwartaln`, `miesieczn`, `tygodnik`, `dziennik`, `czasopism` |
| hun | `evfolyam`, `szam `, `elofizet`, `megjelenik`, `szerkeszt`, `negyedevre`, `havonta`, `hetilap`, `napilap`, `folyoirat` |
| rus | `выходит`, `подписк`, `номер`, `газета`, `журнал`, `редакц`, `ежемесячн`, `еженедельн`, `в год`, `подписная цена` |
| ita | `abbonament`, `esce `, `giornale`, `redazione`, `settimanale`, `mensile`, `trimestr`, `rivista` |
| spa | `suscripcion`, `numero`, `redaccion`, `periodico`, `semanal`, `mensual`, `trimestr`, `revista` |
| dan | `aargang`, `abonnement`, `udkommer`, `redaktion`, `ugeblad`, `maanedlig`, `kvartal`, `tidsskrift` |

#### calibration

| language | stems |
|---|---|
| any | `colorchecker`, `xrite`, `x rite`, `kodak`, `color control`, `gray scale`, `grey scale`, `ektacolor`, `target`, `cyan`, `magenta`, `yellow`, `colour`, `golden thread`, `it8` |

#### music

| language | stems |
|---|---|
| any | `allegro`, `andante`, `moderato`, `adagio`, `presto`, `largo`, `lento`, `vivace`, `piano`, `pianoforte`, `forte`, `cresc`, `cres `, `dim `, `decresc`, `rit `, `rall`, `pizz`, `arco`, `fine `, `coda`, `tempo`, `dolce`, `maestoso`, `andantino`, `allegretto`, `violin`, `soprano`, `alto `, `tenor`, `basso` |
| ces | `klavir`, `zpev`, `pisen`, `housle` |
| deu | `singstimme`, `klavier`, `gesang`, `lied` |
| eng | `voice`, `song` |
| fra | `chant`, `chanson` |
| lat | `cantus`, `missa` |
| pol | `fortepian`, `spiew`, `piesn` |
| hun | `zongora`, `enek` |
| rus | `фортепиано`, `пение`, `романс`, `голос` |
| ita | `canto`, `voce` |
| spa | `canto`, `cancion` |
| dan | `klaver`, `sang ` |

#### map

| language | stems |
|---|---|
| ces | `meritko`, `mapa`, `vrstevnic`, `legenda`, `souradnic`, `poledni`, `severni`, `km `, `kilometr`, `sever` |
| deu | `karte`, `massstab`, `masstab`, `zeichenerklarung`, `nordlich`, `dienstgebrauch`, `planzeiger` |
| eng | `map `, `scale`, `statute miles`, `legend`, `miles`, `north` |
| fra | `carte`, `echelle`, `legende` |
| lat | `mappa`, `tabula`, `milliaria`, `septentrio`, `meridies`, `oriens`, `occidens` |
| pol | `mapa`, `skala`, `legenda` |
| hun | `terkep`, `meretarany`, `jelmagyarazat` |
| rus | `масштаб`, `карта `, `верст`, `условные обозначения` |
| ita | `carta `, `scala`, `legenda` |
| spa | `mapa`, `escala`, `leyenda` |
| dan | `kort `, `malestok` |

#### stamp

| language | stems |
|---|---|
| ces | `knihovna`, `knihovn`, `inv `, `inventar`, `signatura`, `znac `, `majetek`, `razitko` |
| deu | `bibliothek`, `eigentum`, `stempel` |
| eng | `library`, `property of` |
| fra | `bibliotheque`, `propriete` |
| lat | `bibliotheca`, `ex libris`, `exlibris` |
| pol | `bibliotek`, `własnosc` |
| hun | `konyvtar`, `leltar` |
| rus | `библиотек`, `инв ` |
| ita | `biblioteca` |
| spa | `biblioteca` |
| dan | `bibliotek` |

#### table

| language | stems |
|---|---|
| ces | `tab `, `tabulka`, `celkem`, `prumer`, `soucet`, `uhrn`, `pocet`, `poznamka`, `cis ` |
| deu | `tabelle`, `zusammen`, `summe`, `durchschnitt`, `anzahl`, `bemerkung`, `gesamt` |
| eng | `total`, `table `, `average`, `number`, `remarks` |
| fra | `tableau`, `total`, `moyenne`, `nombre`, `observations` |
| lat | `summa`, `tabula`, `numerus` |
| pol | `tabela`, `razem`, `ogołem`, `srednia`, `liczba`, `uwagi` |
| hun | `tablazat`, `osszesen`, `atlag`, `szama`, `megjegyzes` |
| rus | `таблица`, `итого`, `всего`, `средн`, `число`, `количество`, `примечан` |
| ita | `tabella`, `totale`, `numero`, `osservazion` |
| spa | `cuadro`, `total`, `promedio`, `numero`, `observacion` |
| dan | `tabel`, `ialt`, `gennemsnit`, `antal`, `bemærkning` |

#### cover

| language | stems |
|---|---|
| ces | `cena `, `brozovan`, `vazan`, `edice`, `knihovna`, `svazek`, `sesit`, `vyrocni zprava`, `program`, `spisy`, `sbirka` |
| deu | `heft`, `lieferung`, `preis`, `sammlung`, `jahresbericht`, `programm`, `schulnachrichten` |
| eng | `price`, `series`, `annual report` |
| fra | `prix`, `collection`, `livraison`, `fascicule`, `rapport annuel`, `programme` |
| lat | `fasciculus`, `programma` |
| pol | `cena `, `zeszyt`, `seria`, `sprawozdanie`, `program` |
| hun | `fuzet`, `sorozat`, `evkonyv`, `ertesito` |
| rus | `цена`, `выпуск`, `серия`, `отчет`, `программа` |
| ita | `prezzo`, `fascicolo`, `collana`, `annuario`, `programma` |
| spa | `precio`, `coleccion`, `cuaderno`, `anuario`, `memoria anual`, `programa` |
| dan | `pris `, `hefte`, `samling`, `aarsberetning`, `program` |

#### end

| language | stems |
|---|---|
| ces | `konec`, `tiskem`, `vytiskl`, `vytisk` |
| deu | `ende `, `schluss` |
| eng | `the end` |
| fra | `fin `, `acheve d imprimer` |
| lat | `finis`, `explicit`, `laus deo` |
| pol | `koniec` |
| hun | `vege` |
| rus | `конец` |
| ita | `fine ` |
| spa | `fin ` |
| dan | `slut` |
