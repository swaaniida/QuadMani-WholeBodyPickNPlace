"""Exercise exported file:// viewer with network disabled, without the server."""
from pathlib import Path
import tempfile,shutil,json
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
html=ROOT/'reports/ee_ground_compare_20261114/viewer/downloads/as2piper_3d_offline.html'
checks={};errors=[];requests=[]
with tempfile.TemporaryDirectory(prefix='as2-offline-file-') as folder:
    isolated=Path(folder)/html.name;shutil.copy2(html,isolated)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        context=browser.new_context(viewport={'width':1100,'height':780},offline=True)
        page=context.new_page();page.set_default_timeout(45000)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url) if r.url.startswith(('http:','https:')) else None)
        page.goto(isolated.as_uri(),wait_until='load',timeout=90000)
        page.wait_for_function('window.__replayState?.ready',timeout=90000)
        checks['file_protocol']=page.evaluate('location.protocol')=='file:'
        checks['offline_context']=True
        checks['scenarios']=page.locator('#scenario option').evaluate_all('(els)=>els.map(e=>e.value)')
        assert checks['scenarios']==['motion36_vy0','ground64']
        checks['mesh_count']=page.evaluate('window.__replayState.meshCount');assert checks['mesh_count']>20
        page.wait_for_function('window.__replay.stats().triangles>1000')
        checks['render_triangles']=page.evaluate('window.__replay.stats().triangles')
        page.evaluate("document.querySelector('#timeline').value='12';document.querySelector('#timeline').dispatchEvent(new Event('input',{bubbles:true}))")
        page.wait_for_function('window.__replayState.time===12')
        checks['seek']=True
        page.select_option('#checkpoint','0');page.wait_for_function('window.__replayState.selected===0');checks['checkpoint_switch']=True
        page.click('#mode');page.wait_for_function('window.__replayState.compare');checks['four_views']=True
        page.screenshot(path=str(OUT/'offline_four_views.png'))
        page.click('#mode');page.wait_for_function('!window.__replayState.compare')
        before=page.evaluate('window.__replay.getCameras()[0]')
        page.click('[data-view=side]');page.wait_for_timeout(300)
        after=page.evaluate('window.__replay.getCameras()[0]');assert before!=after;checks['camera_preset']=True
        bounds=page.locator('.panel.active').bounding_box();x=bounds['x']+bounds['width']*.5;y=bounds['y']+bounds['height']*.5
        before=page.evaluate('window.__replay.getCameras()[0]');page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+75,y+30,steps=5);page.mouse.up();page.mouse.wheel(0,-180);page.wait_for_timeout(300)
        after=page.evaluate('window.__replay.getCameras()[0]');assert before!=after;checks['orbit_zoom']=True
        page.evaluate("document.querySelector('#opacity').value='.5';document.querySelector('#opacity').dispatchEvent(new Event('input',{bubbles:true}))")
        page.check('#loop');page.select_option('#speed','2');checks['display_controls']=True
        page.click('#play');page.wait_for_function('window.__replayState.time>12.2');page.click('#play');page.wait_for_function('!window.__replayState.playing');checks['play_pause']=True
        page.screenshot(path=str(OUT/'offline_single_view.png'))
        with page.expect_navigation(wait_until='load',timeout=90000):page.select_option('#scenario','ground64')
        page.wait_for_function("window.__replayState?.ready && window.__replayState.scenario==='ground64'",timeout=90000)
        assert page.evaluate('window.__replayState.duration')==64
        page.evaluate("document.querySelector('#timeline').value='18';document.querySelector('#timeline').dispatchEvent(new Event('input',{bubbles:true}))")
        page.wait_for_function('window.__replayState.time===18');checks['ground64_switch_seek']=True
        page.wait_for_function('window.__replay.stats().triangles>1000')
        checks['font_loaded']=page.evaluate("document.fonts.check('14px ReplayKorean')")
        checks['external_requests']=requests;checks['page_errors']=errors
        checks['ground_state']=page.evaluate('window.__replayState')
        assert not errors and not requests and checks['font_loaded']
        browser.close()
(OUT/'browser_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False));print(json.dumps(checks,indent=2,ensure_ascii=False))
