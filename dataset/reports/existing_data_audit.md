# Existing Data Audit

Generated at: 2026-06-25T05:30:16Z

## echo_dot_reddit_reviews.csv
- Records: 31
- Columns: id, dataType, username, title, communityName, body, url, createdAt
- Date range: 2018-02-18T20:24:00Z to 2024-12-19T23:20:18Z
- Duplicates: 0 (0 exact, 0 near)
- URLs: 31 unique, 0 generic, 31 direct post/comment permalinks
- Product identity: 31 normalized/accepted, 0 wrong product, {'accepted': 31}
- Claim quality: 10 firsthand, 1 secondhand, 6 jokes/off-topic/questions/non-customer
- Missing parent context: 0
- Language distribution: {'en': 31}
- Text length: avg 241.45, median 194
- Missing values: `{"body": 0.0, "communityName": 0.0, "createdAt": 0.0, "dataType": 0.0, "id": 0.0, "title": 0.0, "url": 0.0, "username": 0.0}`

## sony_wh1000xm5_reddit_reviews.csv
- Records: 15
- Columns: id, dataType, username, title, communityName, body, url, createdAt, product, source
- Date range: 2024-01-13T13:26:44Z to 2026-06-24T21:42:03Z
- Duplicates: 0 (0 exact, 0 near)
- URLs: 15 unique, 0 generic, 15 direct post/comment permalinks
- Product identity: 11 normalized/accepted, 1 wrong product, {'quarantined_wrong_product': 1, 'accepted': 11, 'quarantined_off_topic': 3}
- Claim quality: 6 firsthand, 0 secondhand, 8 jokes/off-topic/questions/non-customer
- Missing parent context: 0
- Language distribution: {'en': 15}
- Text length: avg 217, median 187
- Missing values: `{"body": 0.0, "communityName": 0.0, "createdAt": 0.0, "dataType": 0.0, "id": 0.0, "product": 0.0, "source": 0.0, "title": 0.0, "url": 0.0, "username": 0.0}`

## wyze_cam_v3_reddit_reviews.csv
- Records: 15
- Columns: id, dataType, username, title, communityName, body, url, createdAt, product, source
- Date range: 2023-05-10T14:22:00Z to 2024-04-22T09:30:00Z
- Duplicates: 0 (0 exact, 0 near)
- URLs: 1 unique, 15 generic, 0 direct post/comment permalinks
- Product identity: 8 normalized/accepted, 0 wrong product, {'accepted': 8, 'quarantined_off_topic': 1, 'quarantined_ambiguous_product': 6}
- Claim quality: 11 firsthand, 0 secondhand, 1 jokes/off-topic/questions/non-customer
- Missing parent context: 6
- Language distribution: {'en': 15}
- Text length: avg 314, median 311
- Missing values: `{"body": 0.0, "communityName": 0.0, "createdAt": 0.0, "dataType": 0.0, "id": 0.0, "product": 0.0, "source": 0.0, "title": 0.0, "url": 0.0, "username": 0.0}`
