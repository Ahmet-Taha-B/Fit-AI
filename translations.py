TRANSLATIONS = {
    "tr": {
        "page_title": "Fitness AI Coach",
        "page_subtitle": "Kişisel fitness ve beslenme asistanınız (LangGraph Hafızalı! 🧠)",
        "api_settings": "🔑 API Ayarları",
        "api_key_label": "Groq API Key",
        "api_key_help": "Groq API anahtarınızı girin (https://console.groq.com/keys)",
        "api_warning": "⚠️ Lütfen API anahtarınızı girin!",
        "api_success": "✅ API anahtarı girildi!",
        "features": "🎯 Özellikler",
        "features_list": """
    ✅ Kişiselleştirilmiş fitness tavsiyeleri
    ✅ Beslenme programları
    ✅ Egzersiz önerileri
    ✅ Bilimsel kaynaklara dayalı
    🧠 **LangGraph Checkpoint Memory**
    """,
        "stats": "📊 Seans İstatistikleri",
        "total_messages": "Toplam Mesaj",
        "your_questions": "Sorularınız",
        "clear_chat": "🗑️ Sohbeti Temizle",
        "powered_by": "Powered by Groq & LangGraph",
        "loading_kb": "📚 Bilgi tabanı yükleniyor...",
        "no_pdfs": "⚠️ data/fitness_pdfs/ klasöründe PDF bulunamadı!",
        "pdfs_loaded": "✅ {count} PDF yüklendi!",
        "vectorstore_error": "❌ Vector store hatası: {error}",
        "retriever_desc": "Fitness ve beslenme bilgilerini içeren PDF'lerden arama yapar. Kullan: egzersiz, beslenme, protein, antrenman soruları için.",
        "system_prompt": """Sen profesyonel bir fitness koçu ve beslenme uzmanısın.

KURALLAR:
- Kısa ve öz cevapla (max 150 kelime)
- Yoga, meditasyon veya ruhani pratiklerden BAHSETME
- Sadece klasik fitness egzersizleri öner (şınav, dambıl, squat, vb.)
- ÖNCEKİ KONUŞMAYI HATIRLA ve takip sorularını akıllıca yanıtla
- "onu", "bunu", "bunlar" gibi referansları önceki mesajlardan anla
- TÜRKÇE CEVAP VER""",
        "api_error": "🔑 Lütfen yan menüden Groq API anahtarınızı girin!",
        "api_info": """
    **Groq API Anahtarı Nasıl Alınır?**
    
    1. [Groq Console](https://console.groq.com/keys) adresine gidin
    2. Hesap oluşturun veya giriş yapın
    3. "API Keys" bölümünden yeni anahtar oluşturun
    4. Anahtarı kopyalayıp yan menüdeki alana yapıştırın
    """,
        "example_questions": "💡 Örnek Sorular",
        "q1": "🏋️ Evde yapabileceğim bir kol antrenmanı öner",
        "q2": "🏃 Kardiyo mu ağırlık mı daha etkili?",
        "q3": "🥗 Kas yapmak için nasıl beslenmem gerekir?",
        "q4": "💪 Günlük kaç protein almalıyım?",
        "q1_text": "Evde yapabileceğim bir kol antrenmanı öner",
        "q2_text": "Kardiyo mu ağırlık mı daha etkili?",
        "q3_text": "Kas yapmak için nasıl beslenmem gerekir?",
        "q4_text": "Günlük kaç protein almalıyım?",
        "thinking": "💭 Düşünüyorum...",
        "agent_error": "❌ Agent oluşturulamadı.",
        "welcome": "👋 **Hoş Geldiniz!** Yukarıdaki örnek sorulardan birini seçin veya aşağıya kendi sorunuzu yazın.",
        "memory_info": """🧠 **LangGraph Checkpoint Memory Aktif!**
    
Otomatik hafıza ile:
- "Peki bunu kaç set yapmalıyım?"
- "Hangi besinlerden alabilirim?"
- "Onu daha detaylı anlat"

gibi takip soruları anlıyorum! 🎯""",
        "chat_placeholder": "💬 Fitness hakkında bir soru sorun...",
        "language": "🌐 Dil / Language",
        "select_language": "Lütfen dilinizi seçin",
        "welcome_title": "Hoş Geldiniz! 💪"
    },
    "en": {
        "page_title": "Fitness AI Coach",
        "page_subtitle": "Your personal fitness and nutrition assistant (With LangGraph Memory! 🧠)",
        "api_settings": "🔑 API Settings",
        "api_key_label": "Groq API Key",
        "api_key_help": "Enter your Groq API key (https://console.groq.com/keys)",
        "api_warning": "⚠️ Please enter your API key!",
        "api_success": "✅ API key entered!",
        "features": "🎯 Features",
        "features_list": """
    ✅ Personalized fitness advice
    ✅ Nutrition plans
    ✅ Exercise recommendations
    ✅ Science-backed information
    🧠 **LangGraph Checkpoint Memory**
    """,
        "stats": "📊 Session Statistics",
        "total_messages": "Total Messages",
        "your_questions": "Your Questions",
        "clear_chat": "🗑️ Clear Chat",
        "powered_by": "Powered by Groq & LangGraph",
        "loading_kb": "📚 Loading knowledge base...",
        "no_pdfs": "⚠️ No PDFs found in data/fitness_pdfs/ folder!",
        "pdfs_loaded": "✅ {count} PDFs loaded!",
        "vectorstore_error": "❌ Vector store error: {error}",
        "retriever_desc": "Searches fitness and nutrition information from PDFs. Use for: exercise, nutrition, protein, workout questions.",
        "system_prompt": """You are a professional fitness coach and nutrition expert.

RULES:
- Keep answers short and concise (max 150 words)
- DO NOT mention yoga, meditation or spiritual practices
- Only suggest classic fitness exercises (push-ups, dumbbells, squats, etc.)
- REMEMBER PREVIOUS CONVERSATION and answer follow-up questions intelligently
- Understand references like "it", "this", "those" from previous messages
- ANSWER IN ENGLISH""",
        "api_error": "🔑 Please enter your Groq API key from the sidebar!",
        "api_info": """
    **How to Get a Groq API Key?**
    
    1. Go to [Groq Console](https://console.groq.com/keys)
    2. Create an account or sign in
    3. Create a new key from "API Keys" section
    4. Copy and paste the key into the sidebar field
    """,
        "example_questions": "💡 Example Questions",
        "q1": "🏋️ Suggest a home arm workout",
        "q2": "🏃 Is cardio or weights more effective?",
        "q3": "🥗 How should I eat to build muscle?",
        "q4": "💪 How much protein should I consume daily?",
        "q1_text": "Suggest a home arm workout",
        "q2_text": "Is cardio or weights more effective?",
        "q3_text": "How should I eat to build muscle?",
        "q4_text": "How much protein should I consume daily?",
        "thinking": "💭 Thinking...",
        "agent_error": "❌ Agent could not be created.",
        "welcome": "👋 **Welcome!** Select one of the example questions above or type your own question below.",
        "memory_info": """🧠 **LangGraph Checkpoint Memory Active!**
    
With automatic memory:
- "How many sets should I do?"
- "Which foods can I get it from?"
- "Explain that in more detail"

I understand follow-up questions like these! 🎯""",
        "chat_placeholder": "💬 Ask a fitness question...",
        "language": "🌐 Language / Dil",
        "select_language": "Please select your language",
        "welcome_title": "Welcome! 💪"
    }
}
