---
layout: single
tags: SQL
category: misc
---


# SQL Murder
I've done many a SQL query in my day, but never to solve a crime! In honor of it being Friday, gonna dust off the old SQL skills and solve one.

Prompt: Investigate a murder that happened on Jan 15, 2018 in SQL City.
`select * from crime_scene_report where type='murder' and city='SQL City' and date='20180115'`

First clue! Two witnesses, one's the last house on Northwestern Drive and the other's an Annabel on Franklin Ave
```SQL
select *
from interview int
inner join person per
 on int.person_id = per.id
 where (per.name like 'Annabel%' and per.address_street_name ='Franklin Ave')
 or (per.address_street_name = 'Northwestern Dr')
 order by address_street_name, address_number desc
 limit 2 -- franklin comes before northwestern so know this will return the 2 accounts
```

Second clue! The killer was at Annabel's gym on 20180109. At the crime scene he had a gold member's Get Fit Now Gym bag with membership number starting 48Z and got into a car with H42W in the plate number.

```SQL
select int.transcript, per.*, fit_mem.id, fit_check.check_in_time, fit_check.check_out_time, fit_check.check_in_date
from get_fit_now_member fit_mem
inner join get_fit_now_check_in fit_check
 on fit_mem.id = fit_check.membership_id
 and fit_mem.membership_status = 'gold'
 and fit_mem.id like '48Z%'
 and fit_check.check_in_date = '20180109'
inner join person per
 on per.id = fit_mem.person_id
inner join drivers_license dri
 on per.license_id = dri.id
 and dri.plate_number like '%H42W%'
inner join interview int
 on int.person_id = per.id
```

"I was hired by a woman with a lot of money. I don't know her name but I know she's around 5'5" (65") or 5'7" (67"). She has red hair and she drives a Tesla Model S. I know that she attended the SQL Symphony Concert 3 times in December 2017."

Well that's a lot of information.

What's "a lot of money" supposed to mean? `select min(annual_income), max(annual_income), avg(annual_income) from income ` yields 10k, 500k, and 53k so gonna take > 53k at the least.

```SQL
select per.id, per.name, dri.age, dri.height, dri.hair_color, dri.gender, dri.car_make, dri.car_model, inc.annual_income
from person per
inner join drivers_license dri
on per.license_id = dri.id
and dri.height between 65 and 67
and dri.gender = 'female'
and dri.hair_color = 'red'
and dri.car_make = 'Tesla' --redundant since only Tesla has Model S
and dri.car_model = 'Model S'
inner join income inc
on inc.ssn = per.ssn
and inc.annual_income > 53000
```

Hmmmm, they both make a lot of money, so let's see who attended the SQL Symphony Concert!
```SQL
select per.id, per.name, dri.age, dri.height, dri.hair_color, dri.gender, dri.car_make, dri.car_model, inc.annual_income
from person per
inner join drivers_license dri
  on per.license_id = dri.id
  and dri.height between 65 and 67
  and dri.gender = 'female'
  and dri.hair_color = 'red'
  and dri.car_make = 'Tesla'
  and dri.car_model = 'Model S'
inner join income inc
  on inc.ssn = per.ssn
  and inc.annual_income > 53000
inner join (
  select person_id, count(*) from facebook_event_checkin
  where event_name = 'SQL Symphony Concert' and date like '201712%'
  group by 1 having count(*) >=3
  ) fb
  on fb.person_id = per.id
```

Miranda Priestly! person_id = 99716.

Unfortunately the browser based game only allows `SELECT` statements so can't test I've got the right answer by doing `INSERT INTO solution VALUES (1, "Miranda Priestly");` into solutions table :(
