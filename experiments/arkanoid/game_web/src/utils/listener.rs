use wasm_bindgen::{prelude::Closure, JsCast};

pub fn window_add_event_listener(
    event_type: &str,
    handler: Box<dyn FnMut(web_sys::Event)>,
    closure: &mut Option<Closure<dyn FnMut(web_sys::Event)>>,
) {
    // wrap rust's closure into js closure, which can be called from js side
    let c = Closure::wrap(handler);

    // register created closure as a handler for DOM event
    web_sys::window()
        .unwrap()
        .add_event_listener_with_callback(event_type, c.as_ref().unchecked_ref())
        .unwrap();

    // save closure so we can unsubscribe later
    *closure = Some(c);
}

pub fn window_remove_event_listener(
    event_type: &str,
    closure: &mut Option<Closure<dyn FnMut(web_sys::Event)>>,
) {
    web_sys::window()
        .unwrap()
        .remove_event_listener_with_callback(
            event_type,
            closure
                .as_ref()
                .expect("closure doesn't exist")
                .as_ref()
                .unchecked_ref(),
        )
        .unwrap();

    // drop closure to free up resources (eg. pointer to game state struct)
    *closure = None;
}
