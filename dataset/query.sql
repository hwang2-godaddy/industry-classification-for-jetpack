with temp_data as
(
select
    B.domain_name,
    A.website_id,
    A.business_category,
    B.gd_domain
from gmode.gocentral_websitebuilder_session A
join gmode.gocentral_website_history B
on A.website_id = B.website_id
where A.business_category not like '%_other_%'
      and B.gd_domain = 0
)
select 
    A.domain_name, 
    A.business_category
from temp_data A,
     (select business_category, count(domain_name) as total_counts from temp_data group by 1) B
where 
A.business_category = B.business_category
and B.total_counts >= 1000  # can delete this line
and A.business_category not in ('',
 'ayesha_curry_food',
 'ayesha_curry_food_2',
 'ayesha_curry_food_3',
 'boxing_day',
 'conference_attendee',
 'custom',
 'dan_nonprofit',
 'dan_nonprofit_2',
 'dan_nonprofit_3',
 'dentallab',
 'elohim_music',
 'elohim_music_2',
 'elohim_music_3',
 'eventphotography',
 'familylawattorney',
 'farms',
 'fathers_day',
 'faxservice',
 'generic',
 'guns_and_ammo',
 'gymnastics',
 'hearingaidproviders',
 'hearingaidrepair',
 'lightning',
 'lyn_fashion',
 'lyn_fashion_2',
 'lyn_fashion_3',
 'meatpacker',
 'na',
 'nine_pin_bowling',
 'other',
 'paloma_florist',
 'paloma_florist_2',
 'paloma_florist_3',
 'personal_fashion',
 'personal_fooddrink',
 'personal_pets',
 'personal_photography',
 'photographers',
 'pianoinstructor',
 'restaurant',
 'taxreturnpreparationfiling',
 'tobaccoshops',
 'travel',
 'trussmanufacturer',
 'tyson_manufacturing',
 'tyson_manufacturing_2',
 'tyson_manufacturing_3',
 'visiting_card')
 and domain_name is not null;