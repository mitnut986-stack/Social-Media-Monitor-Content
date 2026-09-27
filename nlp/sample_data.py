"""
Sample and Benchmark Data for Khmer Social Media Sentiment Analysis.
Contains:
1. TRAINING_DATA: Multi-class labelled dataset for training and fine-tuning ML models.
2. MOCK_YOUTUBE_COMMENTS: Realistic mock data for YouTube videos with timestamps, authors, and likes.
3. MOCK_FACEBOOK_COMMENTS: Realistic mock data for Facebook page posts.
"""

from typing import Any, Dict, List, Tuple

# Curated training data: (Khmer text, Label)
TRAINING_DATA: List[Tuple[str, str]] = [
    # --- POSITIVE EXAMPLES ---
    ("វីដេអូនេះល្អណាស់ អរគុណបងដែលបានចែករំលែកចំណេះដឹង", "Positive"),
    ("សេវាកម្មរហ័សទាន់ចិត្ត និងបុគ្គលិករួសរាយណាស់", "Positive"),
    ("ទំនិញគុណភាពល្អ ដូចការផ្សព្វផ្សាយបេះបិទ ❤️", "Positive"),
    ("ពិតជាអស្ចារ្យមែន ខ្ញុំគាំទ្រ ១០០%", "Positive"),
    ("ចូលចិត្តរបៀបពន្យល់របស់បងណាស់ យល់បានលឿន", "Positive"),
    ("ផលិតផលប្រើត្រូវណាស់ មុខលែងរោលហើយ 👍👍", "Positive"),
    ("តម្លៃសមរម្យ ហើយគុណភាពលើសពីការរំពឹងទុក", "Positive"),
    ("អបអរសាទរចំពោះភាពជោគជ័យរបស់ក្រុមការងារ 🎉", "Positive"),
    ("កោតសរសើរប្អូនស្រីពូកែណាស់ រៀនចប់ដោយជោគជ័យ", "Positive"),
    ("បទចម្រៀងនេះពិរោះរណ្តំចិត្តណាស់ ស្តាប់មិនចេះធុញ", "Positive"),
    ("អរគុណច្រើនបង ជួយដោះស្រាយបញ្ហាខ្ញុំបានហើយ", "Positive"),
    ("ឡូយខ្លាំងណាស់បង បង្កើតវីដេអូបែបនេះច្រើនទៀតមក", "Positive"),
    ("សប្បាយចិត្តណាស់ដែលបានជួបមនុស្សល្អៗបែបនេះ", "Positive"),
    ("កម្មវិធីនេះមានប្រយោជន៍ខ្លាំងណាស់សម្រាប់សិស្សនិស្សិត", "Positive"),
    ("ដឹកជញ្ជូនលឿន វេចខ្ចប់បានស្អាតល្អណាស់ ផ្ដល់ ៥ ផ្កាយ ⭐⭐⭐⭐⭐", "Positive"),
    ("ស្រលាញ់ពេញបេះដូង ជូនពរបងជួបតែសំណាងល្អ 🥰", "Positive"),
    ("ការបកស្រាយក្បោះក្បាយ ច្បាស់លាស់ គួរឲ្យគោរព", "Positive"),
    ("ញាក់សាច់ម៉ងបង អែមណាស់! 🔥🔥", "Positive"),
    ("ឆ្លើយតប chat លឿន ហើយនិយាយផ្អែមល្អណាស់", "Positive"),
    ("មិនដែលខកចិត្តទេពេលទិញហាងបង រាល់ដងពេញចិត្តរហូត", "Positive"),
    ("good job bro, keep it up!", "Positive"),
    ("ពិតជាគំរូល្អសម្រាប់យុវជនជំនាន់ក្រោយ", "Positive"),
    ("អរគុណadminដែលបានឆ្លើយតបយ៉ាងកក់ក្ដៅ", "Positive"),
    ("មាតិកាល្អ ប្លែកភ្នែក មិនសាំញ៉ាំ", "Positive"),
    ("គាំទ្រស្នាដៃខ្មែរ រីកចម្រើនទៅមុខទាំងអស់គ្នា 🇰🇭❤️", "Positive"),
    ("ទិញលើកទី៣ហើយ នៅតែពេញចិត្តដដែល", "Positive"),
    ("ស្រួលប្រើណាស់ មិនស្មុគស្មាញដូចគេថាទេ", "Positive"),
    ("មេដៃធំៗជូនបង 👍 វីដេអូមានគុណភាព", "Positive"),
    ("អស្ចារ្យណាស់ ជោគជ័យធំធេង", "Positive"),
    ("ស្រលាញ់ការងារស្ម័គ្រចិត្តរបស់បងប្អូនណាស់", "Positive"),

    # --- NEGATIVE EXAMPLES ---
    ("សេវាអន់ខ្លាំងណាស់ ចាំតាំងពីព្រឹកទល់ល្ងាចមិនទាន់បាន", "Negative"),
    ("ទំនិញអន់ មិនដូចក្នុងរូបទេ បោកប្រាស់ច្បាស់ណាស់ 😡", "Negative"),
    ("ខកចិត្តណាស់ ផ្ញើឥវ៉ាន់ខូចមកឲ្យភ្ញៀវ 👎", "Negative"),
    ("កុំជឿ ហាងនេះបោកលុយគេទេ", "Negative"),
    ("និយាយកុហកភូតភរទាំងរស់ គ្មានភាពស្មោះត្រង់សោះ", "Negative"),
    ("តម្លៃថ្លៃហួសហេតុ គុណភាពស្មើនឹងសូន្យ", "Negative"),
    ("សេវាកម្មអាក្រក់ បុគ្គលិកសម្តីគ្មានការអប់រំ", "Negative"),
    ("កម្មវិធី error រហូត ពិបាកប្រើណាស់ លុបចោលហើយ", "Negative"),
    ("ខាតទាំងលុយ ខាតទាំងពេលវេលា អត់បានការ", "Negative"),
    ("យឺតដូចអណ្តើក ឆាតទៅ ៣ ថ្ងៃទើបឆ្លើយមួយម៉ាត់", "Negative"),
    ("ស្អប់ណាស់ ធ្វើឲ្យខូចអារម្មណ៍អស់ហើយ", "Negative"),
    ("គ្មានការទទួលខុសត្រូវលើអតិថិជនទាល់តែសោះ", "Negative"),
    ("មិនល្អទេ កុំទិញឲ្យសោះ ប្រយ័ត្នស្ដាយក្រោយ", "Negative"),
    ("មាតិកាគ្មានប្រយោជន៍ ចំណាយពេលមើលឥតប្រយោជន៍", "Negative"),
    ("ចម្លងស្នាដៃគេទាំងស្រុង គ្មានគំនិតច្នៃប្រឌិតផ្ទាល់ខ្លួន", "Negative"),
    ("ដឹកខុសទំនិញ ដល់ពេលសួរដេញដោល block ភ្ញៀវចោលទៀត 🤬", "Negative"),
    ("ធុញណាស់ ព័ត៌មានមិនពិត ក្លែងបន្លំដើម្បីបាន views", "Negative"),
    ("ចំណាយលុយអត់ប្រយោជន៍ គ្មានគុណភាពបន្តិចណាទេ", "Negative"),
    ("សំឡេងរំខានស្ដាប់មិនយល់ទេ វីដេអូអន់ណាស់", "Negative"),
    ("សេវាដឹកជញ្ជូនយឺតយ៉ាវ បែកបាក់ឥវ៉ាន់ខ្ទេច 💔", "Negative"),
    ("worst customer service ever 🤮", "Negative"),
    ("កុំជឿបងប្អូន គេបោកយកលុយកក់រួចគេចាត់ការបាត់", "Negative"),
    ("ពិបាកទាក់ទងណាស់ តេទៅមិនដែលលើក", "Negative"),
    ("មិនសមធ្វើជាអ្នកលក់ទេ ចរិតមិនកែប្រែ", "Negative"),
    ("កុំមើលអី ខាតពេលទទេៗ", "Negative"),
    ("រិះគន់ដើម្បីស្ថាបនា តែលើកនេះអន់ពេកហើយ", "Negative"),
    ("ទិញមកប្រើមិនបាន ៣ ថ្ងៃខូចបាត់ គ្មានធានាទៀត", "Negative"),
    ("ស្អប់អ្នកនិយាយមិនពិត ធ្វើបាបអ្នកគាំទ្រ", "Negative"),
    ("អត់ចូលចិត្តរបៀបនិយាយបែបនេះទេ អត់គួរសម", "Negative"),
    ("គ្មានគុណភាពបន្តិចណាទេ ស្ដាយលុយណាស់", "Negative"),

    # --- NEUTRAL EXAMPLES ---
    ("តើឥវ៉ាន់នេះតម្លៃប៉ុន្មានដែរបង?", "Neutral"),
    ("មានសាខានៅខេត្តសៀមរាបដែរឬទេ?", "Neutral"),
    ("សុំសួរមួយបង តើមានពណ៌សឬខ្មៅទេ?", "Neutral"),
    ("តើវីដេអូភាគក្រោយចេញនៅថ្ងៃណាដែរ?", "Neutral"),
    ("សូមជួយផ្ញើ link ចូលមើលផង admin", "Neutral"),
    ("ទីតាំងហាងនៅម្ដុំណាដែរបង?", "Neutral"),
    ("មានដឹកជញ្ជូនទៅតាមបណ្តាខេត្តទេ?", "Neutral"),
    ("អាចទាក់ទងតាមលេខទូរស័ព្ទណាបានខ្លះ?", "Neutral"),
    ("សូមប្រាប់ពីលក្ខខណ្ឌនៃការទិញបង់រំលស់ផង", "Neutral"),
    ("ពេលណាមានទំនិញចូលស្តុកវិញ?", "Neutral"),
    ("វីដេអូនេះថតនៅទីណាដែរ?", "Neutral"),
    ("មានសៀវភៅនេះជា PDF ដែរទេ?", "Neutral"),
    ("តើត្រូវចុះឈ្មោះតាមវិធីណា?", "Neutral"),
    ("សួស្តីបង ខ្ញុំបានផ្ញើសារចូលប្រអប់សារហើយ", "Neutral"),
    ("អាចប្រាប់ពី specs ម៉ាស៊ីននេះបានទេ?", "Neutral"),
    ("ការប្រជុំនឹងចាប់ផ្ដើមនៅម៉ោង ២ រសៀល", "Neutral"),
    ("ព័ត៌មានលម្អិតសូមទាក់ទងមកកាន់លេខ 012 345 678", "Neutral"),
    ("តើមានការបញ្ចុះតម្លៃក្នុងឱកាសបុណ្យភ្ជុំបិណ្ឌទេ?", "Neutral"),
    ("សូមរង់ចាំការឆ្លើយតបបន្តិច", "Neutral"),
    ("រង់ចាំទស្សនាភាគបន្ត", "Neutral"),
    ("How much is this?", "Neutral"),
    ("Where is your shop located?", "Neutral"),
    ("Please check inbox", "Neutral"),
    ("ម៉ោងប៉ុន្មានបើកហាងបង?", "Neutral"),
    ("សុំឈ្មោះបទភ្លេងខាងក្រោយផង", "Neutral"),
    ("តើត្រូវប្រើជាមួយទូរស័ព្ទ Android ឬ iOS?", "Neutral"),
    ("មានការធានារយៈពេលប៉ុន្មានខែដែរ?", "Neutral"),
    ("ជម្រាបសួរលោកគ្រូ", "Neutral"),
    ("មានលក់ដុំទេបង?", "Neutral"),
    ("សូមអរគុណចំពោះការផ្តល់ព័ត៌មាន", "Neutral"),
]

# Realistic Mock YouTube Comments
MOCK_YOUTUBE_COMMENTS: List[Dict[str, Any]] = [
    {
        "comment_id": "yt_c01",
        "platform": "YouTube",
        "author": "Sokha Mean",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Sokha",
        "comment": "វីដេអូនេះល្អណាស់ ពន្យល់បានក្បោះក្បាយយល់បានលឿន អរគុណបងដែលបានចែករំលែក ❤️❤️",
        "published_at": "2026-09-26T14:32:00Z",
        "like_count": 48,
        "reply_count": 3
    },
    {
        "comment_id": "yt_c02",
        "platform": "YouTube",
        "author": "Rithy Tech",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Rithy",
        "comment": "សំឡេងរំខានខ្លាំងណាស់ ស្ដាប់មិនសូវយល់ទេ ជួយកែសម្រួល microphone ផងបង 👎",
        "published_at": "2026-09-26T15:10:00Z",
        "like_count": 12,
        "reply_count": 1
    },
    {
        "comment_id": "yt_c03",
        "platform": "YouTube",
        "author": "Dara Chan",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Dara",
        "comment": "តើបងប្រើ software អ្វីខ្លះសម្រាប់ edit វីដេអូនេះបង? សុំឈ្មោះផង",
        "published_at": "2026-09-26T16:05:00Z",
        "like_count": 5,
        "reply_count": 2
    },
    {
        "comment_id": "yt_c04",
        "platform": "YouTube",
        "author": "Sreypov Neang",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Sreypov",
        "comment": "ស្រលាញ់ការបង្រៀនរបស់បងណាស់ ច្បាស់លាស់ហើយមានប្រយោជន៍ខ្លាំងណាស់ 👏🎉",
        "published_at": "2026-09-26T16:45:00Z",
        "like_count": 34,
        "reply_count": 0
    },
    {
        "comment_id": "yt_c05",
        "platform": "YouTube",
        "author": "Vannak Gamer",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Vannak",
        "comment": "មាតិកា copy គេមកទាំងដុល គ្មានភាពច្នៃប្រឌិតទាល់តែសោះ អន់ណាស់",
        "published_at": "2026-09-26T17:20:00Z",
        "like_count": 19,
        "reply_count": 4
    },
    {
        "comment_id": "yt_c06",
        "platform": "YouTube",
        "author": "Kosal Angkor",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Kosal",
        "comment": "គាំទ្របងពេញទំហឹង សង្ឃឹមថាឆានែលបងនឹងឡើង 1M Subs ឆាប់ៗនេះ 🇰🇭🔥",
        "published_at": "2026-09-26T18:15:00Z",
        "like_count": 72,
        "reply_count": 2
    },
    {
        "comment_id": "yt_c07",
        "platform": "YouTube",
        "author": "Chenda Pich",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Chenda",
        "comment": "ភាគក្រោយចេញថ្ងៃណាដែរបង? កំពុងទន្ទឹងរង់ចាំទស្សនា",
        "published_at": "2026-09-26T19:00:00Z",
        "like_count": 7,
        "reply_count": 1
    },
    {
        "comment_id": "yt_c08",
        "platform": "YouTube",
        "author": "Bona Khmer",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Bona",
        "comment": "ចំណងជើង clickbait ខ្លាំងណាស់ ដល់ពេលចូលមើលអត់មានអីផង ខាតពេលឥតប្រយោជន៍ 😡",
        "published_at": "2026-09-26T20:30:00Z",
        "like_count": 23,
        "reply_count": 3
    },
    {
        "comment_id": "yt_c09",
        "platform": "YouTube",
        "author": "Piseth Kem",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Piseth",
        "comment": "Great video bro! Very useful information.",
        "published_at": "2026-09-26T21:12:00Z",
        "like_count": 15,
        "reply_count": 0
    },
    {
        "comment_id": "yt_c10",
        "platform": "YouTube",
        "author": "Chanthou Som",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Chanthou",
        "comment": "មិនដែលខកចិត្តទេពេលតាមដានឆានែលបង តែងតែមានគន្លឹះល្អៗជានិច្ច 💯",
        "published_at": "2026-09-26T22:00:00Z",
        "like_count": 41,
        "reply_count": 1
    },
    {
        "comment_id": "yt_c11",
        "platform": "YouTube",
        "author": "Sovann Gaming",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Sovann",
        "comment": "យឺតណាស់បង វីដេអូវែងពេក និយាយដដែលៗគួរឲ្យធុញ",
        "published_at": "2026-09-27T01:15:00Z",
        "like_count": 8,
        "reply_count": 1
    },
    {
        "comment_id": "yt_c12",
        "platform": "YouTube",
        "author": "Thyda Ouk",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Thyda",
        "comment": "សុំ link download source code ផងបង អរគុណច្រើន",
        "published_at": "2026-09-27T02:00:00Z",
        "like_count": 3,
        "reply_count": 0
    },
    {
        "comment_id": "yt_c13",
        "platform": "YouTube",
        "author": "Troll Khmer",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Troll1",
        "comment": "អាល្ងង់ដូចក្របី ធ្វើវីដេអូបានសុទ្ធតែក្បាលខូច មីថោកទាប 🤬",
        "published_at": "2026-09-27T02:30:00Z",
        "like_count": 0,
        "reply_count": 0
    },
    {
        "comment_id": "yt_c14",
        "platform": "YouTube",
        "author": "Scam Bot VIP",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Bot99",
        "comment": "ឈ្នះរង្វាន់ក្លែងក្លាយ! ផ្ទេរលុយមកលេខ 012-888-999 ដើម្បីបើកប្រាក់រង្វាន់ $5,000 ភ្លាមៗ 💸",
        "published_at": "2026-09-27T03:00:00Z",
        "like_count": 1,
        "reply_count": 0
    },
    {
        "comment_id": "yt_c15",
        "platform": "YouTube",
        "author": "Hater 007",
        "author_avatar": "https://api.dicebear.com/7.x/bottts/svg?seed=Hater",
        "comment": "ចាំមើលប្រយ័ត្នងាប់ វ៉ៃបំបែកក្បាលចោល ទៅងាប់ទៅ អាឆ្កែ 😡",
        "published_at": "2026-09-27T03:45:00Z",
        "like_count": 0,
        "reply_count": 1
    }
]


# Realistic Mock Facebook Page Post Comments
MOCK_FACEBOOK_COMMENTS: List[Dict[str, Any]] = [
    {
        "comment_id": "fb_c01",
        "platform": "Facebook",
        "author": "Kagna Heng",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Kagna",
        "comment": "ឥវ៉ាន់ទើបតែមកដល់មិញនេះ ស្អាតខ្លាំងណាស់ គុណភាពល្អឥតខ្ចោះ លើកក្រោយនឹងកុម្ម៉ង់ទៀត ❤️🥰",
        "published_at": "2026-09-26T10:15:00Z",
        "like_count": 18,
        "reply_count": 2
    },
    {
        "comment_id": "fb_c02",
        "platform": "Facebook",
        "author": "Bunthol Ly",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Bunthol",
        "comment": "សេវាដឹកជញ្ជូនអន់ណាស់ ៣ថ្ងៃហើយមិនទាន់ឃើញដឹកមកផង តេទៅអត់លើកទៀត 😡👎",
        "published_at": "2026-09-26T11:00:00Z",
        "like_count": 9,
        "reply_count": 1
    },
    {
        "comment_id": "fb_c03",
        "platform": "Facebook",
        "author": "Sophea Mao",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Sophea",
        "comment": "ម៉ូដនេះមាន size XL អត់បង? តម្លៃប៉ុន្មានដែរ?",
        "published_at": "2026-09-26T11:45:00Z",
        "like_count": 2,
        "reply_count": 1
    },
    {
        "comment_id": "fb_c04",
        "platform": "Facebook",
        "author": "Chhaya Rin",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Chhaya",
        "comment": "បោកប្រាស់គេច្បាស់ណាស់ ផ្ញើឥវ៉ាន់ខុសម៉ូដមក ដល់ពេលសួរដេញដោល block ចោល 🤬",
        "published_at": "2026-09-26T13:30:00Z",
        "like_count": 27,
        "reply_count": 5
    },
    {
        "comment_id": "fb_c05",
        "platform": "Facebook",
        "author": "Bopha Sam",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Bopha",
        "comment": "admin ឆ្លើយតប chat លឿន ហើយរួសរាយរាក់ទាក់ណាស់ ពេញចិត្តសេវាកម្មហាងនេះ 👍✨",
        "published_at": "2026-09-26T14:20:00Z",
        "like_count": 14,
        "reply_count": 1
    },
    {
        "comment_id": "fb_c06",
        "platform": "Facebook",
        "author": "Sambath Keo",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Sambath",
        "comment": "មានទីតាំងហាងផ្ទាល់នៅភ្នំពេញទេបង? ចង់ទៅមើលផ្ទាល់",
        "published_at": "2026-09-26T15:10:00Z",
        "like_count": 4,
        "reply_count": 1
    },
    {
        "comment_id": "fb_c07",
        "platform": "Facebook",
        "author": "Nita Sok",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Nita",
        "comment": "តម្លៃសមរម្យ គុណភាពលេខ១ ប្រើត្រូវខ្លាំងណាស់ ផ្ដល់ ៥ ផ្កាយ ⭐⭐⭐⭐⭐",
        "published_at": "2026-09-26T16:00:00Z",
        "like_count": 32,
        "reply_count": 2
    },
    {
        "comment_id": "fb_c08",
        "platform": "Facebook",
        "author": "Vutha Seng",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Vutha",
        "comment": "ទំនិញថ្លៃហួសហេតុ គុណភាពធម្មតាសោះ លក់កោរភ្ញៀវពេកហើយ",
        "published_at": "2026-09-26T17:40:00Z",
        "like_count": 11,
        "reply_count": 2
    },
    {
        "comment_id": "fb_c09",
        "platform": "Facebook",
        "author": "leakhena Tep",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=leakhena",
        "comment": "Check inbox ផងបង បានផ្ញើសារកុម្ម៉ង់ទៅហើយ",
        "published_at": "2026-09-26T18:25:00Z",
        "like_count": 1,
        "reply_count": 1
    },
    {
        "comment_id": "fb_c10",
        "platform": "Facebook",
        "author": "Panha Nuon",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Panha",
        "comment": "អរគុណច្រើនដែលបានជួយដោះស្រាយបញ្ហាបានទាន់ពេលវេលា សេវាកម្មល្អពិតប្រាកដ 🙏❤️",
        "published_at": "2026-09-26T19:50:00Z",
        "like_count": 16,
        "reply_count": 0
    },
    {
        "comment_id": "fb_c11",
        "platform": "Facebook",
        "author": "Rage Commenter",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Rage",
        "comment": "អាមនុស្សថោកទាប បោកប្រាស់លុយគេច្បាស់ណាស់ កុហកភូតភរទាំងរស់ 🤬👎",
        "published_at": "2026-09-26T20:15:00Z",
        "like_count": 8,
        "reply_count": 2
    },
    {
        "comment_id": "fb_c12",
        "platform": "Facebook",
        "author": "Online Loan Scam",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Loan",
        "comment": "កម្ចីប្រាក់រហ័ស មិនបាច់មានទ្រព្យបញ្ចាំ ផ្ទេរប្រាក់មុនគេចខ្លួន! ឆាតមកតេឡេក្រាមភ្លាម 💸",
        "published_at": "2026-09-26T21:00:00Z",
        "like_count": 0,
        "reply_count": 0
    },
    {
        "comment_id": "fb_c13",
        "platform": "Facebook",
        "author": "Violent Troll",
        "author_avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=TrollFB",
        "comment": "ចុយម្រាយ អាឆ្កួតក្បាលខូច ចាំមើលប្រយ័ត្នងាប់ វ៉ៃបំបែកក្បាល 😡",
        "published_at": "2026-09-26T22:30:00Z",
        "like_count": 0,
        "reply_count": 1
    }
]

