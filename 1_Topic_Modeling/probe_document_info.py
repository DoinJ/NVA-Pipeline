import csv
import sys

import pandas as pd
import argparse


pressreleases_sensitive_topic_name_set = {
    "0_基本法_一帶一路_一國兩制_施政報告",
    "5_殘疾人士_五九九九_culturalservice_ce culturalservice",
    "6_劏房_建築物條例_答覆_主席",
    "9_答覆_主席_他說_教育條例",
    "10_man_arrested_men_crime",
    "11_環境影響評估_協議_答覆_廢物處置條例",
    "15_貧窮線_施政報告_各位嘉賓_在二",
    "35_他說_佔中_答覆_主席",
    "37_中心發言人說_發言人說_調查仍然繼續_此日期前最佳二",
    "40_西布曲明_藥劑業及毒藥條例_衞生署發言人說_昔多芬",
    "46_危險藥物條例_販運危險藥物屬嚴重罪行一經定罪最高可被判罰款五百萬元及終身監禁_危險藥物條例 販運危險藥物屬嚴重罪行一經定罪最高可被判罰款五百萬元及終身監禁_customsgovhk舉報懷疑販毒活動",
    "53_保持香港清潔_政府打擊衞生黑點計劃_全城清潔2015_全城清潔2015 家是香港",
    "108_盧古嘉利說_搖頭丸_她說_友出路",
    "115_更生先鋒計劃_智慧監獄_他說_創藝展更生",
    "120_受限區域_相關宣告_政府亦在 受限區域_政府重申會嚴肅執法",
    "130_任何人士如僱用不可合法受僱的人均屬違法最高罰款為三十五萬元及入獄三年 入境處發言人指出_入境處發言人指出 所有旅客無論受薪與否在未獲入境處處長批准前均不得從事任何僱傭工作違反逗留條件的人士會遭檢控最高罰款為五萬元及入獄兩年_所有旅客無論受薪與否在未獲入境處處長批准前均不得從事任何僱傭工作違反逗留條件的人士會遭檢控最高罰款為五萬元及入獄兩年_入境處發言人指出",
    "158_受限區域_相關宣告_政府發言人表示_若有僱員因宣告而未能上班政府希望僱主能酌情處理不要扣減員工的薪金和福利",
    "187_康文署呼籲泳客顧己及人合力保持泳池清潔衞生落水前不要吃得過飽有需要應先使用洗手間_康文署呼籲泳客顧己及人合力保持泳池清潔衞生落水前不要吃得過飽有需要應先使用洗手間 電台及電視台當值宣布員注意_請盡快播出下列報告並於適當時間重複廣播_電台及電視台當值宣布員注意 請盡快播出下列報告並於適當時間重複廣播",
    "207_防止殘酷對待動物條例_絕育_條例_野生動物保護條例",
    "208_假結婚_入境處發言人重申_入境處發言人重申 本處對非本港居民透過假結婚騙取香港居留的問題一直非常關注並已成立專案小組從不同途徑搜集情報並作出深入調查在取得足夠證據後本處定會對涉案人士提出檢控_本處對非本港居民透過假結婚騙取香港居留的問題一直非常關注並已成立專案小組從不同途徑搜集情報並作出深入調查在取得足夠證據後本處定會對涉案人士提出檢控",
    "250_海牙公約_一國兩制_一帶一路_保全安排",
    "263_公安條例_言論及集會的自由_公眾人士表達訴求時亦必須遵守香港法律和社會秩序_警方重申尊重市民表達意見"
}

'''
bbc_sensitive_topic_name_set = {
    "3_police_man_murder_court",
    "4_ukraine_russia_russian_ukrainian",
    "6_trump_president_donald_donald trump",
    "11_gaza_israel_hamas_israeli",
    "13_police_abuse_sexual_officers",
    "16_iran_syria_iraq_syrian",
    "27_shooting_gun_police_gunman",
    "30_korea_north_north korea_kim",
    "34_china_hong_hong kong_kong",
    "38_police_attack_french_suspect",
    "41_rwanda_asylum_sunak_asylum seekers",
}
'''
bbc_sensitive_topic_name_set = {
    "-1_police_new_ms_need",
    "2_ukraine_russia_russian_ukrainian",
    "4_trump_president_donald_donald trump",
    "5_users_facebook_company_apple",
    "6_police_murder_man_court",
    "7_iran_taliban_afghanistan_syria",
    "8_prince_queen_royal_king",
    "10_brexit_eu_deal_labour",
    "12_gaza_israel_israeli_hamas",
    "14_police_prison_abuse_officers",
    "17_cancer_patients_treatment_women",
    "27_snp_scottish_sturgeon_independence",
    "31_scheme_universal credit_work_universal",
    "32_korea_north_north korea_kim",
    "34_shooting_school_gunman_gun",
    "36_president_venezuela_mexico_maduro",
    "41_china_hong_hong kong_kong",
    "42_police_attack_french_paris",
    "43_hezbollah_lebanon_israel_israeli",
    "48_pay_nhs_strike_nurses",
    "49_johnson_downing_downing street_prime"
}


cnn_sensitive_topic_name_set = {
    "0_says_new_year_said"
}

'''
new_york_times_sensitive_topic_name_set = {
    "-1_said_mr_new_year",
    "2_iraq_iraqi_american_military",
    "4_police_police said_said_man",
    "16_palestinian_israeli_israel_palestinians",
    "31_trade_china_bank_percent",
    "36_russian_russia_putin_chechnya",
    "38_medicare_health_care_drug",
    "47_korea_north_north korea_korean",
    "53_china_chinese_beijing_communist",
    "76_cuba_cuban_elian_castro",
    "92_microsoft_antitrust_judge_court",
    "103_iran_nuclear_uranium_enrichment",
}
'''
new_york_times_sensitive_topic_name_set = {
    "-1_said_mr_new_year",
    "0_ny_beloved_family_wife",
    "3_iraq_iraqi_american_hussein",
    "4_graduated_married_bride_university",
    "5_school_schools_students_education",
    "6_drug_patients_cancer_study",
    "14_palestinian_israeli_israel_palestinians",
    "15_bush_gore_kerry_mr",
    "16_police_police said_said_man",
    "25_building_square_buildings_street",
    "26_ch谩vez_colombia_president_government",
    "28_advertising_worldwide_group_interpublic",
    "29_trade_china_japan_percent",
    "32_trade center_world trade_trade_center",
    "34_taliban_afghanistan_pakistan_afghan",
    "40_tax_social security_bush_budget",
    "41_medicare_health_care_drug",
    "42_russian_russia_putin_chechnya",
    "43_sudan_darfur_nations_united nations",
    "44_energy_oil_environmental_bush",
    "47_aids_africa_infected_virus",
    "49_wine_wines_grapes_grape",
    "50_priests_abuse_bishop_church",
    "51_milosevic_kosovo_mr milosevic_serbian",
    "52_korea_north_north korea_korean",
    "61_chirac_european_berlusconi_minister",
    "63_china_chinese_beijing_communist",
    "69_judge_court_senate_alito",
    "70_florida_election_ballots_gore",
    "71_prison_inmates_offenders_drug",
    "77_plant_indian point_nuclear_indian",
    "78_chess_kramnik_kasparov_pawn",
    "79_death_death penalty_penalty_execution",
    "81_armstrong_tour_tour france_france",
    "84_ireland_northern ireland_northern_irish",
    "86_army_division_sgt_war confirmed",
    "87_police_officers_crime_department",
    "88_smoking_ban_smoke_bars",
    "89_tax_taxes_internal revenue_revenue service",
    "90_holocaust_jews_hitler_german",
    "91_job_business_boss_work",
    "92_china_chinese_taiwan_beijing",
    "95_microsoft_antitrust_judge_court",
    "96_marriage_couples_gay_sex",
    "100_missile_russia_putin_treaty",
    "103_bonds_debt securities_million debt_million",
    "104_pope_john paul_vatican_paul ii",
    "108_liberty_game_points_adubato"
}


zh_mfa_sensitive_topic_name_set = {
    "-1_一带一路_同时_目前_我们认为",
    "0_坚持维护半岛和平稳定_我们认为_据报道_当前形势下",
    "1_4关于第二个问题_中国对南沙群岛及其附近海域拥有无可争辩的主权_关于第一个问题_以实际行动取信于亚洲邻国和国际社会",
    "2_当前_和平共处_近年来_访问期间",
    "3_中国是法治国家_你对此有何评论_涉港_公约",
    "4_台独_世界上只有一个中国_台湾是中国领土不可分割的一部分_台湾是中国的一部分",
    "5_稳定与发展_近年来_关于第一个问题_4关于第二个问题",
    "6_长期_加大外交努力_当前_争取早日取得进展",
    "7_妥善解决_和平_政治解决是叙利亚问题的唯一现实出路_和平 妥善解决",
    "8_中方反对一切形式的恐怖主义_中方坚决反对一切形式的恐怖主义_向遇难者表示哀悼_共同应对恐怖主义威胁"
}

'''
news_peoples_daily_sensitive_topic_name_set = {
    "5_比绍_柬埔寨国家元首柬埔寨民族统一阵线主席诺罗敦_马列_乌尔德",
    "7_七指示_毛主席语录_修养_毛主席万岁",
    "13_1979_国际简讯1979_友好往来1979_据新华社",
    "11_欧佩克_黑石坡煤窑演义_石油输出国组织_石油输出国组织 欧佩克",
    "15_纽约时报_华盛顿邮报_记者张启昕报道_美国",
    "16_新华社_据新华社讯_掌声_据塔斯社莫斯科讯",
    "23_劳动新闻_朝鲜 劳动新闻_记者周必忠报道_记者徐宝康报道",
    "28_华盛顿邮报_纽约时报_洛美协定_一句话新闻1988",
    "33_中俄睦邻友好合作条约_联合国宪章_京都议定书_举报电话",
    "34_毛主席语录1970_毛主席语录1969_毛主席语录1976_毛主席语录1971",
    "35_劳动新闻_朝鲜 劳动新闻_朝鲜_声明说",
    "37_联合国秘书长佩雷斯_记者林皎明报道_德黑兰消息_伊斯兰共和国报",
    "39_圣地亚哥消息_智利总统萨尔瓦多_智利_记者吴惠忠",
    "47_工会法_劳动法_农业法_审计法",
    "48_马约_记者陈特安报道_泰晤士报_本报伦敦电",
    "50_劳动新闻_朝鲜_朝鲜 劳动新闻_民主朝鲜报",
    "60_外事简讯1991_外事简讯1990_外事简讯1993_外事简讯1994",
    "61_记者巴塔尔仓_记者侯耀其_真理报_人民权利报",
    "62_华盛顿消息_纽约消息_伦敦消息_国际简讯1957",
    "79_美国军用飞机侵入我国领空_美国军舰两次侵入我国领海_12第1版_06第1版",
    "86_陆军_仰光消息_锡兰总理西丽玛沃_空军",
    "92_国际简讯1984_19第6版_12第6版_29第6版",
    "97_香港人权法案条例_基本法_香港代议政制_大公报",
    "110_新华社 外事往来1975_外事往来1975_外事往来1976_新华社 外事往来1976",
    "111_争取持久和平争取人民民主_专栏 争取持久和平争取人民民主_新华社 争取持久和平争取人民民主_专栏",
}
'''

news_peoples_daily_sensitive_topic_name_set = {
    "1_记者李长云报道_林峰_12_11",
    "4_美国军用飞机侵入我国领空_美国军用飞机侵入我领空_美机美舰侵入我领空领海_06",
    "5_毛主席语录_毛泽东选集_为人民服务_李志田",
    "6_朱根华_图片_图片 朱根华1988_朱根华1988"
}

gov_xuexiqiangguo_sensitive_topic_name_set = {
    "-1_一带一路_同时_双减_目前",
    "0_一带一路_同时_目前_近年来",
    "1_一国两制_爱国者治港_港独_安全",
    "5_一国两制_一国_两制_一国两制 方针",
}

french_sensitive_topic_name_set = {
        "2_police_policiers_ans_après",
        "3_trump_donald_donald trump_biden",
        "6_présidentielle_droite_gauche_candidat",
        "13_israël_talibans_afghanistan_al",
        "18_ukraine_russie_russe_poutine",
        "35_brexit_royaume uni_uni_royaume",
        "36_mali_sahel_barkhane_président",
        "43_loukachenko_biélorussie_minsk_calédonie",
        "49_violences_victimes_inceste_femmes",
        "53_birmanie_aung_suu_kyi",
        "54_floyd_george floyd_george_police",
        "58_chine_chinois_pékin_xinjiang",
        "60_karabakh_azerbaïdjan_arménie_haut karabakh",
        "81_hongrie_pologne_européen_orban"
}

german_sensitive_topic_name_set = {
    "0_corona_seit_welt_immer"
}

csv.field_size_limit(sys.maxsize)

if __name__ == '__main__':
    # create a parser object
    parser = argparse.ArgumentParser(description="BERTopic main")

    # add arguments
    parser.add_argument("--data_name", type=str, default="pressreleases",
                        choices=[
                            "pressreleases",
                            "gov_xuexiqiangguo", "zh_mfa", "news_peoples_daily",
                            "bbc",
                            "cnn",
                            "new_york_times",
                            'french',
                            "german"
                        ])
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")

    args = parser.parse_args()

    if args.data_name == "pressreleases":
        picked_topic_name_set = pressreleases_sensitive_topic_name_set
    elif args.data_name == "bbc":
        picked_topic_name_set = bbc_sensitive_topic_name_set
    elif args.data_name == "cnn":
        picked_topic_name_set = cnn_sensitive_topic_name_set
    elif args.data_name == "new_york_times":
        picked_topic_name_set = new_york_times_sensitive_topic_name_set
    elif args.data_name == "zh_mfa":
        picked_topic_name_set = zh_mfa_sensitive_topic_name_set
    elif args.data_name == "news_peoples_daily":
        picked_topic_name_set = news_peoples_daily_sensitive_topic_name_set
    elif args.data_name == "gov_xuexiqiangguo":
        picked_topic_name_set = gov_xuexiqiangguo_sensitive_topic_name_set
    elif args.data_name == "german":
        picked_topic_name_set = german_sensitive_topic_name_set
    elif args.data_name == "french":
        picked_topic_name_set = french_sensitive_topic_name_set
    else:
        raise ValueError(f"Unknown data_name: {args.data_name}")

    # 读取数据
    df = pd.read_csv(f"data/{args.data_name}/document_info.csv", engine='python')
    # 查看数据
    print(df.head())
    # 查看数据类型
    print(df.dtypes)
    # 查看数据缺失情况
    print(df.isnull().sum())
    # 查看数据描述性统计
    print(df.describe())

    # 查看敏感主题的文档
    sensitive_document = df[df['Name'].isin(picked_topic_name_set)]
    # Save the sensitive document
    print(sensitive_document.shape)
    # 只保留Document列
    sensitive_document = sensitive_document[['Document']]
    sensitive_document.to_csv(f'data/{args.data_name}/sensitive_document.csv')
