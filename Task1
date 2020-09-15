# Task 1 赛题理解

## 1. 完成天池大赛注册和报名 ✅

## 2. 下载数据 ✅

## 3. 赛题理解
### （1）赛题概况
赛题以金融风控中的个人信贷为背景，要求选手根据贷款申请人的数据信息预测其是否有违约的可能，以此判断是否通过此项贷款，这是一个典型的分类问题。
比赛要求参赛选手根据给定的数据集，建立模型，预测金融风险。这类问题在风控中较为常见，主要就是通过对手中已有的客户信息进行建模和分析来判定客户是否存在潜在违约的风险。
（赛题方同时为本赛题定制了学习方案，其中包括数据科学库、通用流程和baseline方案学习三部分。）

### （2）数据概况
train.csv

* id 为贷款清单分配的唯一信用证标识

* loanAmnt 贷款金额

* term 贷款期限（year）

* interestRate 贷款利率

* installment 分期付款金额

* grade 贷款等级

* subGrade 贷款等级之子级

* employmentTitle 就业职称

* employmentLength 就业年限（年）

* homeOwnership 借款人在登记时提供的房屋所有权状况

* annualIncome 年收入

* verificationStatus 验证状态

* issueDate 贷款发放的月份

* purpose 借款人在贷款申请时的贷款用途类别

* postCode 借款人在贷款申请中提供的邮政编码的前3位数字

* regionCode 地区编码

* dti 债务收入比

* delinquency_2years 借款人过去2年信用档案中逾期30天以上的违约事件数

* ficoRangeLow 借款人在贷款发放时的fico所属的下限范围

* ficoRangeHigh 借款人在贷款发放时的fico所属的上限范围

* openAcc 借款人信用档案中未结信用额度的数量

* pubRec 贬损公共记录的数量

* pubRecBankruptcies 公开记录清除的数量

* revolBal 信贷周转余额合计

* revolUtil 循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额

* totalAcc 借款人信用档案中当前的信用额度总数

* initialListStatus 贷款的初始列表状态

* applicationType 表明贷款是个人申请还是与两个共同借款人的联合申请

* earliesCreditLine 借款人最早报告的信用额度开立的月份

* title 借款人提供的贷款名称

* policyCode 公开可用的策略_代码=1新产品不公开可用的策略_代码=2

* n系列匿名特征 匿名特征n0-n14，为一些贷款人行为计数特征的处理

### （3）评价指标

本次比赛主要采用AUC（Area Under Curve）作为评价指标。指ROC曲线下于坐标轴围成的面积。

ROC（Receiver Operating Characteristic）空间将假正例率（FPR）定义为 X 轴，真正例率（TPR）定义为 Y 轴。
