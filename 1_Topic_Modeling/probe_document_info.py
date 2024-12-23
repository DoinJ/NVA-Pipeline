import csv
import sys

import pandas as pd
import argparse

appledaily_sensitive_topic_name_set = {
    "2_基本法_莫劍弦_林俊謙_一國兩制",
    "20_九二共識_台灣 蘋果日報_台灣_國民黨",
    "6_伊斯蘭國_barackobama_綜合外電報道_華盛頓郵報",
    "30_公安條例_旺角騷亂案_警察通例_大聲公",
    "37_朝鮮日報_南韓 朝鮮日報_北韓_勞動新聞",
    "97_中美貿易戰_wto_中國製造2025_世界貿易組織",
    "112_朝日新聞_日本首相安倍晉三_讀賣新聞_周日",
    "388_限奶令_美贊臣_儲備商品條例_日落條款",
    "586_華為風暴_雙重犯罪_kovrig_michael",
    "702_政府發言人表示_統計處公布_政府發言人稱_英國 脫歐",
    "363_華為風暴_中美貿易戰_華為_763",
    "162_賭博 修訂_賭博_教徒_修訂",
    "110_k仔_俗稱k仔_搖頭丸_毒品",
    "211_平反六四_六四_建設民主中國_毋忘六四",
    "66_公眾衞生_吸煙危害健康_吸煙_公眾衞生 條例",
}

pressreleases_sensitive_topic_name_set = {
    "14_一帶一路_一國兩制_走出去_十二五",
    "20_law_constitutional_basic law_chief",
    "28_他說_條例_主席_答覆",
    "29_基本法_施政報告_一國兩制_記者",
    "43_零毒品_零毒品 的羈押環境_的羈押環境_改良院所設施",
    "48_危險藥物條例_販運危險藥物屬嚴重罪行一經定罪最高可被判罰款五百萬元及終身監禁_危險藥物條例 販運危險藥物屬嚴重罪行一經定罪最高可被判罰款五百萬元及終身監禁_customsgovhk舉報懷疑販毒活動",
    "273_海牙公約_保全安排_一帶一路_一國兩制",
}

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

cnn_sensitive_topic_name_set = {
    "-1_says_new_year_said",
}

new_york_times_sensitive_topic_name_set = {
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

zh_mfa_sensitive_topic_name_set = {
    "-1_一带一路_同时_当前_目前",
    "0_坚持维护半岛和平稳定_我们认为_据报道_当前形势下",
    "1_当前_和平共处_访问期间_近年来",
    "2_以实际行动取信于亚洲邻国和国际社会_钓鱼岛及其附属岛屿自古以来就是中国的固有领土_关于第一个问题_4关于第二个问题",
    "3_中国是法治国家_我们注意到有关报道_涉港_你对此有何评论",
    "4_近年来_稳定与发展_关于第一个问题_访问期间",
    "5_台独_世界上只有一个中国_台湾是中国领土不可分割的一部分_台湾是中国的一部分",
    "6_中方反对一切形式的恐怖主义_中方坚决反对一切形式的恐怖主义_目前_对遇难者表示深切哀悼",
    "7_长期_加大外交努力_照顾彼此关切_我们认为",
    "8_中国对南沙群岛及其附近海域拥有无可争辩的主权_宣言_中国对南海诸岛及其附近海域拥有无可争辩的主权_一贯的",
    "9_妥善解决_和平_4关于第二个问题_和平 妥善解决",
}

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

gov_xuexiqiangguo_sensitive_topic_name_set = {
"1_一带一路_同时_双减_目前",
    "0_一带一路_同时_目前_近年来",
    "1_一国两制_爱国者治港_港独_安全",
    "5_一国两制_一国_两制_一国两制 方针",
}

csv.field_size_limit(sys.maxsize)

if __name__ == '__main__':
    # create a parser object
    parser = argparse.ArgumentParser(description="BERTopic main")

    # add arguments
    parser.add_argument("--data_name", type=str, default="appledaily",
                        choices=[
                            "appledaily", "pressreleases",
                            "gov_xuexiqiangguo", "zh_mfa", "news_peoples_daily",
                            "bbc",
                            "cnn",
                            "new_york_times",
                        ])
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")

    args = parser.parse_args()

    if args.data_name == "appledaily":
        picked_topic_name_set = appledaily_sensitive_topic_name_set
    elif args.data_name == "pressreleases":
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
