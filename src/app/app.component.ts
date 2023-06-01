import { Component } from '@angular/core';
import {HttpClient, HttpHeaders, HttpResponse} from "@angular/common/http";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'dogs';

  showImageHappy: boolean;
  showImageSad: boolean;
  showImageAngry: boolean;
  showImageRelax: boolean;
  showImageX: boolean;
  isHappy: boolean;
  isSad: boolean;
  isAngry: boolean;
  isRelax: boolean;
  imgSelected: any;
  url: any[];

  constructor(private Http: HttpClient) {
    this.showImageHappy = false;
    this.showImageSad = false;
    this.showImageAngry = false;
    this.showImageRelax = false;
    this.showImageX = false;
    this.isHappy = false;
    this.isSad = false;
    this.isAngry = false;
    this.isRelax = false;
    this.imgSelected = false;
    this.url = [];
  }

  uploadFile($event:any) {
    this.destroyImg();
    for(let i = 0; i < $event.target.files.length; i++){
      this.uploadDocument($event.target.files[i], i);
    }
  }

  uploadDocument(file:any, i:number) {
    let fileReader = new FileReader();

    fileReader.onload = (e) => {
      this.url[i] = fileReader.result;
      console.log(this.url);
    }
    
    this.imgSelected = true;
    return fileReader.readAsDataURL(file);
  }

  destroyImg() {
    this.url = [];
    this.imgSelected = false;
  }

  postImg() {
    let request = this.Http.post<String>("http://localhost:3056/cnn", {image:this.url});
    request.subscribe((data => {
      this.isHappy = false;
      this.isSad = false;
      this.isAngry = false;
      this.isRelax = false;

      if(data == 'sad')
        this.isSad = true;
      else if(data == 'relaxed')
        this.isRelax = true;
      else if(data == 'angry')
        this.isAngry = true;
      else if(data == 'happy')
        this.isHappy = true;
      else
        console.log("Não é um Cachorro!")
    }))
  }
}
