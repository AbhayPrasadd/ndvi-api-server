from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
from datetime import date, timedelta
import os

# Initialize Earth Engine using Service Account
service_account = 'ndvi-service@coherent-coder-454119-u9.iam.gserviceaccount.com'  
credentials = ee.ServiceAccountCredentials(service_account, 'service-account.json')
ee.Initialize(credentials)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/ndvi', methods=['GET'])
def get_ndvi():
    lat = float(request.args.get('lat'))
    lng = float(request.args.get('lng'))

    point = ee.Geometry.Point([lng, lat])
    buffer = point.buffer(100)

    today = date.today()
    start_date = today - timedelta(days=15)
    end_date = today

    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(point)
                  .filterDate(str(start_date), str(end_date))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

    if collection.size().getInfo() == 0:
        return jsonify({'error': 'No recent cloud-free Sentinel-2 image found for this location.'})

    image = collection.median()
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndvi_image = ndvi.clip(buffer)

    stats = ndvi_image.reduceRegion(
        reducer=ee.Reducer.mean()
                .combine(ee.Reducer.min(), '', True)
                .combine(ee.Reducer.max(), '', True),
        geometry=buffer,
        scale=10,
        maxPixels=1e9
    ).getInfo()

    poor_mask = ndvi_image.lt(0.2)
    moderate_mask = ndvi_image.gte(0.2).And(ndvi_image.lte(0.5))
    good_mask = ndvi_image.gt(0.5)

    poor_count = poor_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=buffer,
        scale=10,
        maxPixels=1e9
    ).getInfo().get('NDVI', 0)

    moderate_count = moderate_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=buffer,
        scale=10,
        maxPixels=1e9
    ).getInfo().get('NDVI', 0)

    good_count = good_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=buffer,
        scale=10,
        maxPixels=1e9
    ).getInfo().get('NDVI', 0)

    total = poor_count + moderate_count + good_count or 1

    breakdown = {
        'poor_percent': round((poor_count / total) * 100, 2),
        'moderate_percent': round((moderate_count / total) * 100, 2),
        'good_percent': round((good_count / total) * 100, 2)
    }

    return jsonify({
        'lat': lat,
        'lng': lng,
        'ndvi_mean': round(stats.get('NDVI_mean', 0), 4),
        'ndvi_min': round(stats.get('NDVI_min', 0), 4),
        'ndvi_max': round(stats.get('NDVI_max', 0), 4),
        'ndvi_breakdown': breakdown,
        'date_range': f'{start_date} to {end_date}'
    })

@app.route('/ndvi-trend', methods=['GET'])
def get_ndvi_trend():
    lat = float(request.args.get('lat'))
    lng = float(request.args.get('lng'))
    weeks = int(request.args.get('weeks', 4))

    point = ee.Geometry.Point([lng, lat])
    buffer = point.buffer(100)
    today = date.today()

    trend_data = []

    for i in range(weeks):
        end = today - timedelta(days=i * 7)
        start = end - timedelta(days=7)

        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(point)
                      .filterDate(str(start), str(end))
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

        if collection.size().getInfo() == 0:
            trend_data.append({
                'week': f'{start.strftime("%b %d")}–{end.strftime("%b %d")}',
                'ndvi': None,
                'thumb_url': None
            })
            continue

        image = collection.median()
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI').clip(buffer)

        stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        ndvi_val = round(stats.get('NDVI', 0), 4) if stats.get('NDVI') else None

        thumb_url = ndvi.getThumbURL({
            'region': buffer,
            'dimensions': 512,
            'format': 'png',
            'min': 0.0,
            'max': 1.0,
            'palette': ['brown', 'yellow', 'green']
        })

        trend_data.append({
            'week': f'{start.strftime("%b %d")}–{end.strftime("%b %d")}',
            'ndvi': ndvi_val,
            'thumb_url': thumb_url
        })

    return jsonify({
        'lat': lat,
        'lng': lng,
        'trend': trend_data
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
