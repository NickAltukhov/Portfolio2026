# Внесение данных в таблицу
/*
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/supermarket_sales.csv'
INTO TABLE supermarket_sales
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'           
LINES TERMINATED BY '\n'    
IGNORE 1 ROWS             
(`Invoice ID`,Branch,City,`Customer type`,Gender,`Product line`,`Unit price`,Quantity,`Tax 5%`,Total,@date_var,Time,Payment,cogs,`gross margin percentage`,`gross income`,Rating)  
SET Date = STR_TO_DATE(@date_var, '%c/%e/%Y');
*/


# Сумма продаж и средний рейтинг по филиалам
SELECT 
	Branch,
    COUNT(Branch) AS row_counts,
	SUM(Total) AS sum_total,
    AVG(Total) AS avg_total,
	AVG(Rating) AS avg_rating
FROM supermarket_sales
GROUP BY Branch
ORDER BY sum_total DESC
LIMIT 100;
# При том, что количество строк в филиале C немного меньше, суммарные продажи у него больше

# Сравним продажи между клиентами с картой лояльности и без, в разрезе категорий товара
SELECT
	`Product line`,
    `Customer type`,
	SUM(Total) AS sum_total,
	AVG(`Unit price`) AS avg_unit_price,
    COUNT(`Product line`) AS row_counts
FROM supermarket_sales
GROUP BY `Product line`, `Customer type`
ORDER BY `Product line`, `Customer type`
LIMIT 100;
/* 
Можно сделать вывод, что еду и напитки люди значительно чаще покупают с картой лояльности
А вот электронику, наоборот, значительно чаще покупают без карты лояльности

Люди покупают еду и напитки часто, поэтому даже небольшая скидка от карты лояльности
со временем приносит ощутимую экономию

Электронику покупают значительно реже, чем продукты. Из-за этого карта лояльности
не кажется такой выгодной – скидка не ощущается как регулярная экономия.
И человек не уверен, что в следующий раз придет за покупкой электроники именно в этот супермаркет.
А продукты, наоборот, часто покупают в одном месте.
*/


# Определение дней недели и месяца, когда больше всего продаж
WITH days_month_week AS (
	SELECT 
		DAY(Date) AS  day_of_month,
        DAYNAME(Date) AS day_of_week,
        HOUR(Time) AS hour_of_day,
        Total
    FROM supermarket_sales
),
total_by_d_of_m AS (
	SELECT 
		SUM(Total) AS sum_total,
        day_of_month
    FROM days_month_week
    GROUP BY day_of_month
),
total_by_d_of_w AS (
	SELECT 
		SUM(Total) AS sum_total,
		day_of_week
    FROM days_month_week
    GROUP BY day_of_week
),
total_by_hour AS (
	SELECT 
		SUM(Total) AS sum_total,
		hour_of_day
    FROM days_month_week
    GROUP BY hour_of_day
)

SELECT 
    (SELECT day_of_month FROM total_by_d_of_m ORDER BY sum_total DESC LIMIT 1) AS best_day_of_month,
    (SELECT day_of_week FROM total_by_d_of_w ORDER BY sum_total DESC LIMIT 1) AS best_day_of_week,
    (SELECT hour_of_day FROM total_by_hour ORDER BY sum_total DESC LIMIT 1) AS best_hour;
# Логично видеть субботу в самом успешном дне недели в продажах супермаркета
# Самый успешный день месяца - 15ый
# Самое успешное время - 19 часов

# Топ-5 самых больших покупок для каждого филиала
WITH ranks_by_branch AS (SELECT
	Total,
    Branch,
	DENSE_RANK() OVER (PARTITION BY Branch ORDER BY Total DESC) AS rank_by_branch
FROM supermarket_sales)

SELECT * 
FROM ranks_by_branch
WHERE rank_by_branch <= 5
ORDER BY Total DESC
LIMIT 100;
# Самые большие значения представлены в большей степени филиалом C

# Зависимость между средним чеком и рейтингами клиентов
SELECT
	AVG(Total) AS avg_total,
    SUM(Total) AS sum_total,
    CASE 
        WHEN Rating < 5 THEN '0-5'
        WHEN Rating >= 5 AND Rating < 7 THEN '5-7'
        WHEN Rating >= 7 AND Rating < 9 THEN '7-9'
        ELSE '9+'
    END AS rating_range
FROM supermarket_sales
GROUP BY rating_range
ORDER BY avg_total DESC
LIMIT 100;
# Интересно, самый большой чек получился у людей, которые ниже всего оценили процесс покупок/обслуживания
# Возможно это как раз связано с тем, что они не ожидали такой итоговой суммы покупки

# Вычисление кумулятивной суммы Total
SELECT
	Branch,
    Date,
    Time,
	SUM(Total) OVER (PARTITION BY Branch ORDER BY Date, Time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumsum
FROM supermarket_sales
WHERE Branch IN ('A', 'B')
LIMIT 1000;

