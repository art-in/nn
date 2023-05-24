use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::JsCast;

pub fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .expect("failed to get window")
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("failed to register `requestAnimationFrame` callback");
}

pub fn start_request_animation_frame_loop(mut f: Box<dyn FnMut() -> bool>) {
    // allocate closure on the heap since (a) we need it to live longer than
    // current function, otherwise it would be dropped on return, and (b) we
    // need to save pointer to it inside itself, in order to run in loop
    let closure = Rc::new(RefCell::new(None));

    // clone pointer so we can move and save one inside closure, and use another
    // one later to call RAF for the first time
    let closure_clone = closure.clone();

    // here we're creating closure that will save pointer to itself, which forms
    // standard circular reference situation: closure is not dropped because
    // it's referenced by Rc pointer, and Rc pointer is not dropped because it's
    // referenced from closure
    *closure.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let should_continue = f();

        if !should_continue {
            // drop pointer to closure so entire chain of associated resources
            // can be dropped as well: pointer -> closure -> f -> f's resources
            *closure_clone.borrow_mut() = None;
            return;
        }

        request_animation_frame(closure_clone.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(closure.borrow().as_ref().unwrap());
}
