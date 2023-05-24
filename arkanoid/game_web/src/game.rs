use arkanoid_game::game::{Game, GameStatus};
use std::{cell::RefCell, ops::Deref, rc::Rc};
use wasm_bindgen::{prelude::Closure, JsCast};
use web_sys::HtmlCanvasElement;

use crate::{
    drawer::CanvasGameStateDrawer,
    utils::{
        canvas::resize_canvas,
        listener::{window_add_event_listener, window_remove_event_listener},
        raf::start_request_animation_frame_loop,
        size::Size,
    },
};

pub struct GameWeb {
    canvas: HtmlCanvasElement,
    game: Game<CanvasGameStateDrawer>,

    // closures are saved so we can unsubscribe later and free up resources.
    // unsubscribing never really happens right now as game loops forever,
    // so we could simply .forget() them and do not save anything, but i leave
    // it anyway just as an example of more mature approach
    on_resize: Option<Closure<dyn FnMut(web_sys::Event)>>,
    on_pointermove: Option<Closure<dyn FnMut(web_sys::Event)>>,
}

impl GameWeb {
    pub fn new(canvas: HtmlCanvasElement) -> Self {
        GameWeb {
            canvas: canvas.clone(),
            game: Game::new(CanvasGameStateDrawer::new(canvas)),
            on_resize: None,
            on_pointermove: None,
        }
    }
}

// 1. create wrapper-type, since game only used through pointer.
//    use custom type instead of type alias to be able to impl methods for it
// 2. allocate game on the heap because it should outlive main() and be
//    referenced from RAF loop and event handler closures
// 3. game is reference counted pointer (Rc) and not unique pointer (Box)
//    because we need to clone it to all the closures
pub struct GameWebRc(Rc<RefCell<GameWeb>>);

impl GameWebRc {
    pub fn new(canvas: HtmlCanvasElement) -> Self {
        GameWebRc(Rc::new(RefCell::new(GameWeb::new(canvas))))
    }
}

// implement deref to avoid ".0" on every access to its only field
impl Deref for GameWebRc {
    type Target = Rc<RefCell<GameWeb>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// override clone so returned typed is correct GameRc and not Rc<..>
impl Clone for GameWebRc {
    fn clone(&self) -> Self {
        GameWebRc(self.0.clone())
    }
}

impl GameWebRc {
    pub fn start() {
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document
            .query_selector("canvas")
            .unwrap()
            .expect("failed to find canvas element")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

        let game_web = GameWebRc::new(canvas);

        game_web.on_resize();
        game_web.subscribe();

        // start game loop
        start_request_animation_frame_loop(Box::new(move || game_web.step()));
    }

    pub fn step(&self) -> bool {
        let status = self.borrow_mut().game.step();
        self.borrow().game.draw();

        if status == GameStatus::InProgress {
            true
        } else {
            self.unsubscribe();
            false
        }
    }

    fn subscribe(&self) {
        let game = self.clone();
        window_add_event_listener(
            "resize",
            Box::new(move |_| game.on_resize()),
            &mut self.borrow_mut().on_resize,
        );

        let game = self.clone();
        window_add_event_listener(
            "pointermove",
            Box::new(move |event| {
                game.on_pointermove(event.dyn_into::<web_sys::PointerEvent>().unwrap())
            }),
            &mut self.borrow_mut().on_pointermove,
        );
    }

    fn unsubscribe(&self) {
        window_remove_event_listener("resize", &mut self.borrow_mut().on_resize);
        window_remove_event_listener("pointermove", &mut self.borrow_mut().on_pointermove);
    }

    fn on_resize(&self) {
        let window = web_sys::window().unwrap();
        let body = window.document().unwrap().body().unwrap();

        let target_css_size = Size {
            width: body.client_width() as f64,
            height: body.client_height() as f64,
        };

        resize_canvas(
            &self.borrow().canvas,
            target_css_size,
            window.device_pixel_ratio(),
        )
        .unwrap();
    }

    fn on_pointermove(&self, event: web_sys::PointerEvent) {
        let window = web_sys::window().unwrap();
        let body = window.document().unwrap().body().unwrap();

        let virtual_x = event.client_x() as f64 / body.client_width() as f64;

        self.borrow_mut().game.move_platform_to(virtual_x);
    }
}
