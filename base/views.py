from django.shortcuts import render
from django.http import JsonResponse
from django.views.generic import FormView
from .forms import *
import pickle
import simplemma


import pytesseract
from PIL import Image

def home(request):
    return render(request, "home.html")

def modellabel(request):
    return render(request, "modellabel.html")
  
def modeltext(request):
    return render(request, "modeltext.html")

def getPredictions(nalehani, lukrativni, hyperlink, priloha, vizual, gramatika, finance):
    model = pickle.load(open('ml_model.sav', 'rb'))
    scaled = pickle.load(open('scaler.sav', 'rb'))

    prediction = model.predict(scaled.transform([
        [nalehani, lukrativni, hyperlink, priloha, vizual, gramatika, finance]
    ]))
    
    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'
    
def image_to_text(request):
    if request.method == 'POST':
        image = request.FILES['image']
        image = Image.open(image)
        text = pytesseract.image_to_string(image, lang ="ces")
        textedited = text.replace("\n", " ").replace("\r", " ").replace(".","").replace(",","").split()
    
        stopwords = ['a', 'aby', 'ahoj', 'aj', 'ale', 'anebo', 'ani', 'aniž', 'ano', 'asi', 'aspoň', 'atd', 'atp', 'az', 'ačkoli', 'až', 'bez', 'beze', 'blízko', 'bohužel', 'brzo', 'bude', 'budem', 'budeme', 'budes', 'budete', 'budeš', 'budou', 'budu', 'by', 'byl', 'byla', 'byli', 'bylo', 'byly', 'bys', 'byt', 'být', 'během', 'chce', 'chceme', 'chcete', 'chceš', 'chci', 'chtít', 'chtějí', "chut'", 'chuti', 'ci', 'clanek', 'clanku', 'clanky', 'co', 'coz', 'což', 'cz', 'daleko', 'dalsi', 'další', 'den', 'deset', 'design', 'devatenáct', 'devět', 'dnes', 'do', 'dobrý', 'docela', 'dva', 'dvacet', 'dvanáct', 'dvě', 'dál', 'dále', 'děkovat', 'děkujeme', 'děkuji', 'email', 'ho', 'hodně', 'i', 'jak', 'jakmile', 'jako', 'jakož', 'jde', 'je', 'jeden', 'jedenáct', 'jedna', 'jedno', 'jednou', 'jedou', 'jeho', 'jehož', 'jej', 'jeji', 'jejich', 'její', 'jelikož', 'jemu', 'jen', 'jenom', 'jenž', 'jeste', 'jestli', 'jestliže', 'ještě', 'jež', 'ji', 'jich', 'jimi', 'jinak', 'jine', 'jiné', 'jiz', 'již', 'jsem', 'jses', 'jseš', 'jsi', 'jsme', 'jsou', 'jste', 'já', 'jí', 'jím', 'jíž', 'jšte', 'k', 'kam', 'každý', 'kde', 'kdo', 'kdy', 'kdyz', 'když', 'ke', 'kolik', 'kromě', 'ktera', 'ktere', 'kteri', 'kterou', 'ktery', 'která', 'které', 'který', 'kteři', 'kteří', 'ku', 'kvůli', 'ma', 'mají', 'mate', 'me', 'mezi', 'mi', 'mit', 'mne', 'mnou', 'mně', 'moc', 'mohl', 'mohou', 'moje', 'moji', 'možná', 'muj', 'musí', 'muze', 'my', 'má', 'málo', 'mám', 'máme', 'máte', 'máš', 'mé', 'mí', 'mít', 'mě', 'můj', 'může', 'na', 'nad', 'nade', 'nam', 'napiste', 'napište', 'naproti', 'nas', 'nasi', 'načež', 'naše', 'naši', 'ne', 'nebo', 'nebyl', 'nebyla', 'nebyli', 'nebyly', 'nechť', 'nedělají', 'nedělá', 'nedělám', 'neděláme', 'neděláte', 'neděláš', 'neg', 'nejsi', 'nejsou', 'nemají', 'nemáme', 'nemáte', 'neměl', 'neni', 'není', 'nestačí', 'nevadí', 'nez', 'než', 'nic', 'nich', 'nimi', 'nove', 'novy', 'nové', 'nový', 'nula', 'ná', 'nám', 'námi', 'nás', 'náš', 'ní', 'ním', 'ně', 'něco', 'nějak', 'někde', 'někdo', 'němu', 'němuž', 'o', 'od', 'ode', 'on', 'ona', 'oni', 'ono', 'ony', 'osm', 'osmnáct', 'pak', 'patnáct', 'po', 'pod', 'podle', 'pokud', 'potom', 'pouze', 'pozdě', 'pořád', 'prave', 'pravé', 'pred', 'pres', 'pri', 'pro', 'proc', 'prostě', 'prosím', 'proti', 'proto', 'protoze', 'protože', 'proč', 'prvni', 'první', 'práve', 'pta', 'pět', 'před', 'přede', 'přes', 'přese', 'při', 'přičemž', 're', 'rovně', 's', 'se', 'sedm', 'sedmnáct', 'si', 'sice', 'skoro', 'smí', 'smějí', 'snad', 'spolu', 'sta', 'sto', 'strana', 'sté', 'sve', 'svych', 'svym', 'svymi', 'své', 'svých', 'svým', 'svými', 'svůj', 'ta', 'tady', 'tak', 'take', 'takhle', 'taky', 'takze', 'také', 'takže', 'tam', 'tamhle', 'tamhleto', 'tamto', 'tato', 'te', 'tebe', 'tebou', "ted'", 'tedy', 'tema', 'ten', 'tento', 'teto', 'ti', 'tim', 'timto', 'tipy', 'tisíc', 'tisíce', 'to', 'tobě', 'tohle', 'toho', 'tohoto', 'tom', 'tomto', 'tomu', 'tomuto', 'toto', 'trošku', 'tu', 'tuto', 'tvoje', 'tvá', 'tvé', 'tvůj', 'ty', 'tyto', 'téma', 'této', 'tím', 'tímto', 'tě', 'těm', 'těma', 'těmu', 'třeba', 'tři', 'třináct', 'u', 'určitě', 'uz', 'už', 'v', 'vam', 'vas', 'vase', 'vaše', 'vaši', 've', 'vedle', 'večer', 'vice', 'vlastně', 'vsak', 'vy', 'vám', 'vámi', 'vás', 'váš', 'více', 'však', 'všechen', 'všechno', 'všichni', 'vůbec', 'vždy', 'z', 'za', 'zatímco', 'zač', 'zda', 'zde', 'ze', 'zpet', 'zpravy', 'zprávy', 'zpět', 'čau', 'či', 'článek', 'článku', 'články', 'čtrnáct', 'čtyři', 'šest', 'šestnáct', 'že']
        
        def lemmatizeandstop(inputtext):          
            textstopped = []
            finaltext = []
            
            for i in inputtext:
                inner_list = []
                if i not in stopwords:
                    inner_list.append(simplemma.lemmatize(i, lang='cs').lower())  
                print(inner_list)
                textstopped.append(inner_list)
                
            for i in textstopped:
                if len(i) > 0:
                    finaltext.append(' '.join(i))
            
            return(finaltext)
                
        textlemmastop = lemmatizeandstop(textedited)
                
        return render(request, 'modeltext.html', {'text': text, "textlemmastop": textlemmastop})
            
    return render(request, 'modeltext.html')

def result(request):
    nalehani = str(request.GET.get('nalehani', 0))
    lukrativni = str(request.GET.get('lukrativni', 0))
    hyperlink = str(request.GET.get('hyperlink', 0))
    priloha = str(request.GET.get('priloha', 0))
    vizual = str(request.GET.get('podezrela', 0))
    gramatika = str(request.GET.get('gramatika', 0))
    finance = str(request.GET.get('finance', 0))
    
    if nalehani == "yes":
        nalehani = 1
    else:
        nalehani = 0
        
    if lukrativni == "yes":
        lukrativni = 1
    else:
        lukrativni = 0
    
    if hyperlink == "yes":
        hyperlink = 1
    else:
        hyperlink = 0
    
    if priloha == "yes":
        priloha = 1
    else:
        priloha = 0
        
    if vizual == "yes":
        vizual = 1
    else:
        vizual = 0
        
    if gramatika == "yes":
        gramatika = 1
    else: 
        gramatika = 0
    
    if finance == "yes":
        finance = 1
    else: 
        finance = 0
        
    result = getPredictions(nalehani, lukrativni, hyperlink, priloha, vizual, gramatika, finance)

    return render(request, 'result.html', {'result': result})

class HomeView(FormView):
    form_class = UploadForm
    template_name = 'index.html'
    success_url = '/'

    def form_valid(self, form):
        upload = self.request.FILES['file']
        return super().form_valid(form)
